const express = require('express');
const multer = require('multer');
const fs = require('fs');
const cors = require('cors');
const { HfInference } = require('@huggingface/inference');

require('dotenv').config();

const app = express();
const upload = multer({ dest: 'uploads/' });
const hf = new HfInference(process.env.HF_TOKEN);

app.use(cors({
  origin: ['http://localhost:3000', 'https://mood-e.vercel.app'],
  methods: ['GET', 'POST']
}

));


// Analyze Face Route
app.post('/api/analyze-face', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    // Read the uploaded file
    const imageBuffer = fs.readFileSync(req.file.path);

    // Detect emotions
    const emotionResult = await hf.imageClassification({
      model: 'dima806/facial_emotions_image_detection',
      data: imageBuffer
    });
    const emotions = emotionResult.reduce((acc, item) => {
      acc[item.label] = item.score;
      return acc;
    }, {});

    // Detect gender
    const genderResult = await hf.imageClassification({
      model: 'rizvandwiki/gender-classification',
      data: imageBuffer
    });
    const gender = genderResult.reduce((acc, item) => {
      acc[item.label] = item.score;
      return acc;
    }, {});

    // Estimate age
    const ageResult = await hf.imageClassification({
      model: 'nateraw/vit-age-classifier',
      data: imageBuffer
    });
    const age = ageResult.reduce((acc, item) => {
      acc[item.label] = item.score;
      return acc;
    }, {});

    // Detect mask
    const maskResult = await hf.imageClassification({
      model: 'Hemg/Face-Mask-Detection',
      //model: "Docty/nose-mask-classification",
      data: imageBuffer
    });
    const mask = maskResult.reduce((acc, item) => {
      acc[item.label] = item.score;
      return acc;
    }, {});


    fs.unlinkSync(req.file.path);


    res.json({
      emotions,
      gender,
      age,
      mask
    });
  } catch (error) {
    console.error('Error analyzing face:', error);
    res.status(500).json({ error: 'Analysis failed' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
