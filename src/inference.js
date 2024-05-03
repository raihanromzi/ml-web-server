/*
Inference machine learning merujuk pada proses menggunakan model yang telah dilatih untuk
membuat prediksi dan menghasilkan output dari data yang belum pernah dilihat dalam proses training atau pelatihan model.
*/
import tfjs from "@tensorflow/tfjs-node";

/**
 * Loads a TensorFlow.js model from the specified file URL.
 *
 * @return {Promise<tf.LayersModel>} A Promise that resolves to the loaded model.
 */
function loadModel() {
  const modelUrl = "file://models/model.json";
  return tfjs.loadLayersModel(modelUrl);
}

/**
 * Predicts the output of a given model using an image buffer.
 *
 * @param {Object} model - The TensorFlow.js model to use for prediction.
 * @param {Buffer} imageBuffer - The buffer containing the image data.
 * @return {Array} The predicted output of the model.
 */
function predict(model, imageBuffer) {
  const tensor = tfjs.node
    // melakukan decoding dari gambar dengan format JPEG dalam buffer (karena buffer dapat di proses TF)
    .decodeJpeg(imageBuffer)
    // mengubah gambar yang sebelumnya di-decode menjadi 150 x 150 piksel dengan algoritma nearest neighbor
    .resizeNearestNeighbor([150, 150])
    // preprocessing -> menambahkan dimensi ekstra pada tensor serta mengonversi nilai-nilai dalam tensor tersebut ke tipe data float.
    .expandDims()
    .toFloat();

  return model.predict(tensor).data();
}

export { loadModel, predict };
