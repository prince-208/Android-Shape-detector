# Shape Detection Challenge

## Overview

This project implements a **geometric shape detection algorithm** in TypeScript using **Node.js,json,js,html**.  
It identifies and classifies shapes such as **triangles**, **rectangles**, **squares**, and **circles** from input images.

## 🚀 Features
- Detects and classifies **basic geometric shapes** from images  
- Uses **Node.js,json,js,html** for image processing and contour analysis  
- Works with any standard image file (PNG, JPG, JPEG)  
- Lightweight and modular TypeScript implementation  
- Easy to extend with new shape types  

## 🧠 Approach Overview

1. **Image Preprocessing**  
   - Convert to grayscale  
   - Apply Gaussian blur to reduce noise  
   - Perform Canny edge detection 

2. **Contour Detection**
   - Identify shape boundaries using `cv.findContours()`

3. **Polygon Approximation**
   - Approximate each contour using `cv.approxPolyDP()` to determine the number of sides

4. **Classification**
   - **3 sides → Triangle**  
   - **4 sides → Square / Rectangle (based on aspect ratio)**  
   - **>4 sides → Circle / Polygon (based on circularity)**

5. **Output**
   - Displays shape names in the console  


## Setup Instructions

### Prerequisites

- Node.js (version 16 or higher)
- npm or yarn package manager

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Project Structure

```
shape-detector/
├── src/
│   ├── main.ts          # Main application code (implement here)
│   └── style.css        # Basic styling
├── test-images/         # Test images directory
├── expected_results.json # Expected detection results
├── index.html          # Application UI
└── README.md           # This file
```

## Challenge Requirements

### Primary Task

Implement the `detectShapes()` method in the `ShapeDetector` class located in `src/main.ts`. This method should:

1. Analyze the provided `ImageData` object
2. Detect all geometric shapes present in the image
3. Classify each shape into one of the five required categories
4. Return detection results with specified format

### Implementation Location

```typescript
// File: src/main.ts
async detectShapes(imageData: ImageData): Promise<DetectionResult> {
  // TODO: Implement your shape detection algorithm here
  // This is where you write your code
}
```


## Test Images

The `test-images/` directory contains 10 test images with varying complexity:

1. **Simple shapes** - Clean, isolated geometric shapes
2. **Mixed scenes** - Multiple shapes in single image
3. **Complex scenarios** - Overlapping shapes, noise, rotated shapes
4. **Edge cases** - Very small shapes, partial occlusion
5. **Negative cases** - Images with no detectable shapes

See `expected_results.json` for detailed expected outcomes for each test image.






