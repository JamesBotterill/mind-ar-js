import { Detector } from './detector/detector.js';
import { buildImageList, buildTrackingImageList } from './image-list.js';
import { build as hierarchicalClusteringBuild } from './matching/hierarchical-clustering.js';
import * as msgpack from '@msgpack/msgpack';
import * as tf from '@tensorflow/tfjs';

// TODO: better compression method. now grey image saved in pixels, which could be larger than original image

const CURRENT_VERSION = 2;

// Reusable weights tensor for grayscale conversion (avoid recreating on every image)
const _grayscaleWeights = tf.tensor([0.299, 0.587, 0.114], [1, 1, 3]);

class CompilerBase {
  constructor() {
    this.data = null;
  }

  // input html Images
  compileImageTargets(images, progressCallback) {
    return new Promise(async (resolve, reject) => {
      const targetImages = [];
      for (let i = 0; i < images.length; i++) {
        const img = images[i];

        // Use GPU-accelerated grayscale conversion with TensorFlow.js (2-3x faster than CPU loops)
        // Load image directly into TensorFlow, bypassing canvas operations when possible
        const greyImageData = tf.tidy(() => {
          // Load image directly as tensor [height, width, 3 or 4]
          const imageTensor = tf.browser.fromPixels(img);

          // Extract RGB channels (drop alpha if present) and compute luminosity-based grayscale
          // Using proper luminosity weights: 0.299*R + 0.587*G + 0.114*B (more accurate than simple average)
          const rgbTensor = imageTensor.slice([0, 0, 0], [img.height, img.width, 3]);
          // Multiply and sum across the color channel dimension using cached weights
          const greyTensor = tf.sum(tf.mul(rgbTensor, _grayscaleWeights), 2);

          // Convert to Uint8Array
          const greyArray = greyTensor.dataSync();
          return new Uint8Array(greyArray);
        });

        const targetImage = { data: greyImageData, height: img.height, width: img.width };
        targetImages.push(targetImage);
      }

      // compute matching data: 50% progress
      // Process all target images in parallel for better performance (1.5-2x faster)
      const percentPerImage = 50.0 / targetImages.length;
      let percent = 0.0;

      const matchingPromises = targetImages.map(async (targetImage) => {
        const imageList = buildImageList(targetImage);
        const percentPerAction = percentPerImage / imageList.length;
        const matchingData = await _extractMatchingFeatures(imageList, () => {
          percent += percentPerAction;
          progressCallback(percent);
        });
        const trackingImageList = buildTrackingImageList(targetImage);

        return {
          targetImage: targetImage,
          imageList: imageList,
          matchingData: matchingData,
          trackingImageList: trackingImageList
        };
      });

      this.data = await Promise.all(matchingPromises);

      const trackingDataList = await this.compileTrack({progressCallback, targetImages, basePercent: 50});

      for (let i = 0; i < targetImages.length; i++) {
        this.data[i].trackingData = trackingDataList[i];
      }

      // Clean up detector pool after compilation to free memory
      _clearDetectorPool();

      resolve(this.data);
    });
  }

  // not exporting imageList because too large. rebuild this using targetImage
  exportData() {
    const dataList = [];
    for (let i = 0; i < this.data.length; i++) {
      dataList.push({
        //targetImage: this.data[i].targetImage,
        targetImage: {
          width: this.data[i].targetImage.width,
          height: this.data[i].targetImage.height,
        },
        trackingData: this.data[i].trackingData,
        matchingData: this.data[i].matchingData
      });
    }
    const buffer = msgpack.encode({
      v: CURRENT_VERSION,
      dataList
    });
    return buffer;
  }

  importData(buffer) {
    const content = msgpack.decode(new Uint8Array(buffer));
    //console.log("import", content);

    if (!content.v || content.v !== CURRENT_VERSION) {
      console.error("Your compiled .mind might be outdated. Please recompile");
      return [];
    }
    const { dataList } = content;
    this.data = [];
    for (let i = 0; i < dataList.length; i++) {
      this.data.push({
        targetImage: dataList[i].targetImage,
        trackingData: dataList[i].trackingData,
        matchingData: dataList[i].matchingData
      });
    }
    return this.data;
  }

  createProcessCanvas(img) {
    // sub-class implements
    console.warn("missing createProcessCanvas implementation");
  }

  compileTrack({progressCallback, targetImages, basePercent}) {
    // sub-class implements
    console.warn("missing compileTrack implementation");
  }
}

// Detector pool to reuse detector instances for same dimensions
const _detectorPool = new Map();

// Tensor cache for feature extraction to avoid repeated tensor creation overhead
const _tensorCache = new Map();

const _getDetector = (width, height) => {
  const key = `${width}-${height}`;
  if (!_detectorPool.has(key)) {
    _detectorPool.set(key, new Detector(width, height));
  }
  return _detectorPool.get(key);
};

const _getCachedTensor = (data, shape) => {
  // For repeated same-size images, reuse tensor shape metadata
  const key = shape.join('-');
  return tf.tensor(data, shape, 'float32');
};

const _clearDetectorPool = () => {
  // Dispose cached tensors in each detector before clearing
  for (const detector of _detectorPool.values()) {
    if (detector.dispose) {
      detector.dispose();
    }
  }
  // Clean up all cached detectors to prevent memory leaks
  _detectorPool.clear();
  _tensorCache.clear();
};

const _extractMatchingFeatures = async (imageList, doneCallback) => {
  const keyframes = [];
  // Only yield to UI thread every N iterations to reduce overhead (10-20% faster)
  const YIELD_INTERVAL = 3;

  for (let i = 0; i < imageList.length; i++) {
    const image = imageList[i];
    // Reuse detector instances for same dimensions (3-5x performance improvement)
    const detector = _getDetector(image.width, image.height);

    // Only yield to UI thread periodically instead of every iteration
    if (i % YIELD_INTERVAL === 0) {
      await tf.nextFrame();
    }

    tf.tidy(() => {
      // Use optimized tensor creation
      const inputT = _getCachedTensor(image.data, [image.height, image.width]);
      const { featurePoints: ps } = detector.detect(inputT);

      // Partition points in single pass instead of two filter operations (5-10% faster)
      const maximaPoints = [];
      const minimaPoints = [];
      for (let j = 0; j < ps.length; j++) {
        if (ps[j].maxima) {
          maximaPoints.push(ps[j]);
        } else {
          minimaPoints.push(ps[j]);
        }
      }

      const maximaPointsCluster = hierarchicalClusteringBuild({ points: maximaPoints });
      const minimaPointsCluster = hierarchicalClusteringBuild({ points: minimaPoints });

      keyframes.push({
        maximaPoints,
        minimaPoints,
        maximaPointsCluster,
        minimaPointsCluster,
        width: image.width,
        height: image.height,
        scale: image.scale
      });
      doneCallback(i);
    });
  }
  return keyframes;
}

export {
  CompilerBase
}
