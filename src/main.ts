import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

export interface Point {
  x: number;
  y: number;
}

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM TO IMPLEMENT
   * Method for detecting shapes in an image
   * @param imageData - ImageData from canvas
   * @returns Promise<DetectionResult> - Detection results
   *
   * TODO: Implement shape detection algorithm here
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const startTime = performance.now();

    const shapes: DetectedShape[] = [];
    const { data, width, height } = imageData;

    // Step 1: Convert to binary image (threshold)
    const binary = this.createBinaryImage(data, width, height);
    
    // Step 2: Find contours (connected components)
    const contours = this.findContours(binary, width, height);
    
    // Step 3: Process each contour to detect shapes
    for (const contour of contours) {
      if (contour.length < 3) continue; // Need at least 3 points
      
      // Simplify contour using Douglas-Peucker algorithm
      const simplified = this.simplifyContour(contour, 2.0);
      
      if (simplified.length < 3) continue;
      
      // Classify shape
      const shapeInfo = this.classifyShape(simplified, contour);
      
      if (shapeInfo) {
        shapes.push(shapeInfo);
      }
    }

    const processingTime = performance.now() - startTime;

    return {
      shapes,
      processingTime,
      imageWidth: width,
      imageHeight: height,
    };
  }

  /**
   * Convert image to binary using threshold
   */
  private createBinaryImage(data: Uint8ClampedArray, width: number, height: number): boolean[] {
    const binary = new Array(width * height).fill(false);
    const threshold = 128; // Threshold for black/white separation
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const gray = (r + g + b) / 3;
      const index = Math.floor(i / 4);
      binary[index] = gray < threshold; // true for dark pixels (shapes)
    }
    
    return binary;
  }

  /**
   * Find contours using flood fill algorithm
   */
  private findContours(binary: boolean[], width: number, height: number): Point[][] {
    const visited = new Array(width * height).fill(false);
    const contours: Point[][] = [];
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const index = y * width + x;
        if (binary[index] && !visited[index]) {
          const contour = this.floodFill(binary, visited, x, y, width, height);
          if (contour.length > 10) { // Filter small noise
            contours.push(contour);
          }
        }
      }
    }
    
    return contours;
  }

  /**
   * Flood fill to extract connected component and its boundary
   */
  private floodFill(
    binary: boolean[],
    visited: boolean[],
    startX: number,
    startY: number,
    width: number,
    height: number
  ): Point[] {
    const allPoints: Point[] = [];
    const boundaryPoints: Point[] = [];
    const stack: Point[] = [{ x: startX, y: startY }];
    
    while (stack.length > 0) {
      const { x, y } = stack.pop()!;
      const index = y * width + x;
      
      if (x < 0 || x >= width || y < 0 || y >= height || visited[index] || !binary[index]) {
        continue;
      }
      
      visited[index] = true;
      allPoints.push({ x, y });
      
      // Check if this is a boundary point (has at least one white neighbor)
      let isBoundary = false;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            isBoundary = true;
          } else {
            const nIndex = ny * width + nx;
            if (!binary[nIndex]) {
              isBoundary = true;
            }
          }
        }
      }
      
      if (isBoundary) {
        boundaryPoints.push({ x, y });
      }
      
      // Check 8-connected neighbors
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          stack.push({ x: x + dx, y: y + dy });
        }
      }
    }
    
    // Use boundary points if available, otherwise use all points
    // Sort boundary points to form a continuous contour
    if (boundaryPoints.length > 0) {
      return this.orderBoundaryPoints(boundaryPoints, allPoints);
    }
    
    return allPoints;
  }

  /**
   * Order boundary points to form a continuous contour
   */
  private orderBoundaryPoints(boundaryPoints: Point[], allPoints: Point[]): Point[] {
    if (boundaryPoints.length === 0) return allPoints;
    
    // Start with the first boundary point
    const ordered: Point[] = [boundaryPoints[0]];
    const remaining = new Set(boundaryPoints.slice(1).map(p => `${p.x},${p.y}`));
    
    let current = boundaryPoints[0];
    
    while (remaining.size > 0) {
      let nearest: Point | null = null;
      let minDist = Infinity;
      
      for (const pointStr of remaining) {
        const [x, y] = pointStr.split(',').map(Number);
        const dist = Math.abs(x - current.x) + Math.abs(y - current.y);
        if (dist < minDist && dist <= 2) { // Only consider nearby points
          minDist = dist;
          nearest = { x, y };
        }
      }
      
      if (nearest) {
        ordered.push(nearest);
        remaining.delete(`${nearest.x},${nearest.y}`);
        current = nearest;
      } else {
        // If no nearby point found, pick the closest one
        for (const pointStr of remaining) {
          const [x, y] = pointStr.split(',').map(Number);
          const dist = Math.sqrt(Math.pow(x - current.x, 2) + Math.pow(y - current.y, 2));
          if (dist < minDist) {
            minDist = dist;
            nearest = { x, y };
          }
        }
        if (nearest) {
          ordered.push(nearest);
          remaining.delete(`${nearest.x},${nearest.y}`);
          current = nearest;
        } else {
          break;
        }
      }
    }
    
    return ordered.length > 0 ? ordered : allPoints;
  }

  /**
   * Simplify contour using Douglas-Peucker algorithm
   */
  private simplifyContour(contour: Point[], epsilon: number): Point[] {
    if (contour.length <= 2) return contour;
    
    // Find the point with maximum distance from line between first and last
    let maxDist = 0;
    let maxIndex = 0;
    const first = contour[0];
    const last = contour[contour.length - 1];
    
    for (let i = 1; i < contour.length - 1; i++) {
      const dist = this.pointToLineDistance(contour[i], first, last);
      if (dist > maxDist) {
        maxDist = dist;
        maxIndex = i;
      }
    }
    
    if (maxDist > epsilon) {
      // Recursively simplify
      const left = this.simplifyContour(contour.slice(0, maxIndex + 1), epsilon);
      const right = this.simplifyContour(contour.slice(maxIndex), epsilon);
      return [...left.slice(0, -1), ...right];
    } else {
      return [first, last];
    }
  }

  /**
   * Calculate distance from point to line segment
   */
  private pointToLineDistance(point: Point, lineStart: Point, lineEnd: Point): number {
    const A = point.x - lineStart.x;
    const B = point.y - lineStart.y;
    const C = lineEnd.x - lineStart.x;
    const D = lineEnd.y - lineStart.y;
    
    const dot = A * C + B * D;
    const lenSq = C * C + D * D;
    let param = -1;
    
    if (lenSq !== 0) param = dot / lenSq;
    
    let xx: number, yy: number;
    
    if (param < 0) {
      xx = lineStart.x;
      yy = lineStart.y;
    } else if (param > 1) {
      xx = lineEnd.x;
      yy = lineEnd.y;
    } else {
      xx = lineStart.x + param * C;
      yy = lineStart.y + param * D;
    }
    
    const dx = point.x - xx;
    const dy = point.y - yy;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Classify shape based on simplified contour
   */
  private classifyShape(simplified: Point[], originalContour: Point[]): DetectedShape | null {
    const numVertices = simplified.length;
    
    // Calculate bounding box and center
    const bounds = this.calculateBounds(originalContour);
    const center = {
      x: bounds.minX + bounds.width / 2,
      y: bounds.minY + bounds.height / 2,
    };
    
    // Calculate area
    const area = this.calculateArea(originalContour);
    
    // Calculate circularity
    const perimeter = this.calculatePerimeter(simplified);
    const circularity = (4 * Math.PI * area) / (perimeter * perimeter);
    
    // Classify based on number of vertices and geometric properties
    let shapeType: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
    let confidence = 0.8;
    
    if (circularity > 0.85 && numVertices >= 8) {
      // High circularity indicates a circle
      shapeType = "circle";
      confidence = Math.min(0.95, 0.7 + circularity * 0.25);
    } else if (numVertices === 3 || (numVertices >= 3 && numVertices <= 5 && this.isTriangle(simplified))) {
      shapeType = "triangle";
      confidence = 0.85 + (numVertices === 3 ? 0.1 : -0.1);
    } else if (numVertices === 4 || (numVertices >= 4 && numVertices <= 6 && this.isRectangle(simplified))) {
      shapeType = "rectangle";
      confidence = 0.9 + (numVertices === 4 ? 0.08 : -0.1);
    } else if (numVertices === 5 || (numVertices >= 5 && numVertices <= 7 && this.isPentagon(simplified))) {
      shapeType = "pentagon";
      confidence = 0.85 + (numVertices === 5 ? 0.1 : -0.1);
    } else if (numVertices >= 8 && numVertices <= 12 && this.isStar(simplified)) {
      shapeType = "star";
      confidence = 0.8;
    } else if (numVertices >= 6 && numVertices <= 8) {
      // Could be a pentagon with some simplification
      shapeType = "pentagon";
      confidence = 0.75;
    } else {
      // Unknown shape, skip
      return null;
    }
    
    return {
      type: shapeType,
      confidence: Math.min(1.0, confidence),
      boundingBox: {
        x: bounds.minX,
        y: bounds.minY,
        width: bounds.width,
        height: bounds.height,
      },
      center,
      area,
    };
  }

  /**
   * Check if shape is a triangle
   */
  private isTriangle(points: Point[]): boolean {
    if (points.length !== 3) return false;
    // Check if angles are reasonable for a triangle
    const angles = this.calculateAngles(points);
    const minAngle = Math.min(...angles);
    const maxAngle = Math.max(...angles);
    return minAngle > 20 && maxAngle < 160; // Reasonable triangle angles
  }

  /**
   * Check if shape is a rectangle
   */
  private isRectangle(points: Point[]): boolean {
    if (points.length < 4) return false;
    
    // Check if angles are close to 90 degrees
    const angles = this.calculateAngles(points);
    let rightAngles = 0;
    for (const angle of angles) {
      if (Math.abs(angle - 90) < 30) rightAngles++;
    }
    
    // Should have at least 2-3 right angles for a rectangle
    return rightAngles >= 2;
  }

  /**
   * Check if shape is a pentagon
   */
  private isPentagon(points: Point[]): boolean {
    if (points.length < 5) return false;
    
    // Check if it's roughly regular (similar edge lengths)
    const edgeLengths = this.calculateEdgeLengths(points);
    const avgLength = edgeLengths.reduce((a, b) => a + b, 0) / edgeLengths.length;
    const variance = edgeLengths.reduce((sum, len) => sum + Math.pow(len - avgLength, 2), 0) / edgeLengths.length;
    const stdDev = Math.sqrt(variance);
    
    // Low variance indicates regular pentagon
    return stdDev / avgLength < 0.3;
  }

  /**
   * Check if shape is a star
   */
  private isStar(points: Point[]): boolean {
    if (points.length < 8) return false;
    
    // Stars have alternating convex/concave vertices
    // Check for alternating pattern in angles
    const angles = this.calculateAngles(points);
    let alternating = 0;
    for (let i = 0; i < angles.length - 1; i++) {
      const diff = Math.abs(angles[i] - angles[i + 1]);
      if (diff > 30) alternating++;
    }
    
    // Stars typically have many alternating angles
    return alternating >= points.length / 2;
  }

  /**
   * Calculate angles at each vertex
   */
  private calculateAngles(points: Point[]): number[] {
    const angles: number[] = [];
    const n = points.length;
    
    for (let i = 0; i < n; i++) {
      const prev = points[(i - 1 + n) % n];
      const curr = points[i];
      const next = points[(i + 1) % n];
      
      const v1 = { x: prev.x - curr.x, y: prev.y - curr.y };
      const v2 = { x: next.x - curr.x, y: next.y - curr.y };
      
      const dot = v1.x * v2.x + v1.y * v2.y;
      const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
      const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
      
      const cosAngle = dot / (mag1 * mag2);
      const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle))) * (180 / Math.PI);
      angles.push(angle);
    }
    
    return angles;
  }

  /**
   * Calculate edge lengths
   */
  private calculateEdgeLengths(points: Point[]): number[] {
    const lengths: number[] = [];
    for (let i = 0; i < points.length; i++) {
      const next = points[(i + 1) % points.length];
      const dx = next.x - points[i].x;
      const dy = next.y - points[i].y;
      lengths.push(Math.sqrt(dx * dx + dy * dy));
    }
    return lengths;
  }

  /**
   * Calculate bounding box
   */
  private calculateBounds(points: Point[]): { minX: number; minY: number; width: number; height: number } {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    for (const point of points) {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    }
    
    return {
      minX,
      minY,
      width: maxX - minX + 1,
      height: maxY - minY + 1,
    };
  }

  /**
   * Calculate area using shoelace formula
   */
  private calculateArea(points: Point[]): number {
    let area = 0;
    const n = points.length;
    
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += points[i].x * points[j].y;
      area -= points[j].x * points[i].y;
    }
    
    return Math.abs(area) / 2;
  }

  /**
   * Calculate perimeter
   */
  private calculatePerimeter(points: Point[]): number {
    let perimeter = 0;
    for (let i = 0; i < points.length; i++) {
      const next = points[(i + 1) % points.length];
      const dx = next.x - points[i].x;
      const dy = next.y - points[i].y;
      perimeter += Math.sqrt(dx * dx + dy * dy);
    }
    return perimeter;
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById(
      "originalCanvas"
    ) as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById(
      "testImages"
    ) as HTMLDivElement;
    this.evaluateButton = document.getElementById(
      "evaluateButton"
    ) as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById(
      "evaluationResults"
    ) as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(
      this.detector,
      this.evaluateButton,
      this.evaluationResultsDiv
    );

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) {
        await this.processImage(file);
      }
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";

      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);

      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${
              shape.type.charAt(0).toUpperCase() + shape.type.slice(1)
            }</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(
          1
        )})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html +=
        "<p>No shapes detected. Please implement the detection algorithm.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      const module = await import("./test-images-data.js");
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      // Add upload functionality as first grid item
      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName
          .replace(/[_-]/g, " ")
          .replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}" 
               onclick="loadTestImage('${imageName}', '${dataUrl}')" 
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const file = new File([blob], name, { type: "image/svg+xml" });

          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);

          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (
        event: MouseEvent,
        imageName: string
      ) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      // Add upload functionality
      (window as any).triggerFileUpload = () => {
        this.imageInput.click();
      };
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Run 'node convert-svg-to-png.js' to generate test image data.</p>
        <p>SVG files are available in the test-images/ directory.</p>
      `;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ShapeDetectionApp();
});
