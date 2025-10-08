/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, Modality, GenerateContentResponse } from '@google/genai';

// --- Type Declarations ---
declare const google: any;
declare const gapi: any;

interface CanvasImage {
  id: number;
  name: string;
  img: HTMLImageElement;
  x: number;
  y: number;
  width: number;
  height: number;
  isVisible: boolean;
  isLocked: boolean;
}
type Handle = 'nw' | 'ne' | 'sw' | 'se';
type WorkspaceState = 'editor' | 'loading' | 'output';

// --- Constants ---
const API_KEY = process.env.API_KEY;
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
const DRIVE_SCOPES = 'https://www.googleapis.com/auth/drive.readonly';
const MAX_CANVAS_DISPLAY_WIDTH = 680; // Max width in pixels for the canvas element
const HANDLE_SIZE = 16; // Logical pixel size for handles
const HANDLE_COLOR = '#4f46e5';
const HANDLE_STROKE = '#ffffff';

// --- Global State ---
const ai = new GoogleGenAI({ apiKey: API_KEY });
const driveFiles = new Map<string, File[]>();
let lastGeneratedImageDataUrl: string | null = null;

// --- UI Elements ---
const modelSelect = document.getElementById('modelSelect') as HTMLSelectElement;
const generateButton = document.getElementById(
  'generateButton',
) as HTMLButtonElement;
const surpriseMeButton = document.getElementById(
  'surpriseMeButton',
) as HTMLButtonElement;
const clearCanvasButton = document.getElementById(
  'clearCanvasButton',
) as HTMLButtonElement;

// Workspace State Containers
const editorWrapper = document.getElementById(
  'editor-wrapper',
) as HTMLDivElement;
const loadingContainer = document.getElementById(
  'loading-container',
) as HTMLDivElement;
const outputWrapper = document.getElementById(
  'output-wrapper',
) as HTMLDivElement;

// Output elements
const textOutput = document.getElementById('output') as HTMLPreElement;
const imageOutputContainer = document.getElementById(
  'image-output-container',
) as HTMLDivElement;
const textOutputContainer = document.getElementById(
  'text-output-container',
) as HTMLDivElement;
const outputImage = document.getElementById('outputImage') as HTMLImageElement;

// Action Buttons
const downloadImageButton = document.getElementById(
  'downloadImageButton',
) as HTMLButtonElement;
const downloadTextButton = document.getElementById(
  'downloadTextButton',
) as HTMLButtonElement;
const textBackToEditorButton = document.getElementById(
  'textBackToEditorButton',
) as HTMLButtonElement;
const imageBackToEditorButton = document.getElementById(
  'imageBackToEditorButton',
) as HTMLButtonElement;
const addToCanvasButton = document.getElementById(
  'addToCanvasButton',
) as HTMLButtonElement;
const addLayerButton = document.getElementById(
  'addLayerButton',
) as HTMLButtonElement;
const duplicateLayerButton = document.getElementById(
  'duplicateLayerButton',
) as HTMLButtonElement;
const deleteLayerButton = document.getElementById(
  'deleteLayerButton',
) as HTMLButtonElement;

// Text Prompts
const globalPrompt = document.getElementById(
  'globalPrompt',
) as HTMLTextAreaElement;
const faceDetails = document.getElementById(
  'faceDetails',
) as HTMLTextAreaElement;
const dressPose = document.getElementById('dressPose') as HTMLTextAreaElement;
const background = document.getElementById('background') as HTMLTextAreaElement;

// Image Options
const aspectRatioSelect = document.getElementById(
  'aspectRatioSelect',
) as HTMLSelectElement;
const generationOptionsContainer = document.getElementById(
  'generation-options-container',
) as HTMLDivElement;
const imageEditUpload = document.getElementById(
  'image-edit-upload',
) as HTMLDivElement;
const referenceImageSections = document.querySelectorAll(
  '.reference-image-section',
) as NodeListOf<HTMLDetailsElement>;
const imageFilePreview = document.getElementById(
  'imageFilePreview',
) as HTMLDivElement;
const editorMainView = document.getElementById(
  'editor-main-view',
) as HTMLDivElement;
const outpaintCanvas = document.getElementById(
  'outpaintCanvas',
) as HTMLCanvasElement;
const canvasCtx = outpaintCanvas.getContext('2d');
const layerList = document.getElementById('layerList') as HTMLUListElement;
const layerPanel = document.getElementById(
  'layer-panel-container',
) as HTMLDivElement;
const splitter = document.getElementById('splitter') as HTMLDivElement;
const canvasSizeLabel = document.getElementById(
  'canvas-size-label',
) as HTMLDivElement;

// File Inputs
const imageFileInput = document.getElementById('imageFile') as HTMLInputElement;
const ALL_FILE_INPUT_IDS = [
  'imageFile',
  'boyFaceImageFile',
  'girlFaceImageFile',
  'dressPoseImageFile',
  'backgroundImageFile',
];

// --- Canvas State ---
let canvasImages: CanvasImage[] = [];
let activeImageId: number | null = null;
let isDragging = false;
let isResizing = false;
let dragStart = { x: 0, y: 0 };
let activeHandle: Handle | null = null;

// --- Utility Functions ---
/**
 * Converts a file to a base64 encoded string.
 */
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror = (error) => reject(error);
  });
}

/**
 * Creates a blank, transparent canvas of a given size and returns it as a base64 string.
 * This is used to force the Gemini model to generate an image with specific dimensions.
 * @param width The desired width of the canvas.
 * @param height The desired height of the canvas.
 * @returns A promise that resolves to the base64 encoded PNG string.
 */
async function createBlankCanvasAsBase64(
  width: number,
  height: number,
): Promise<string> {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  // A new canvas is transparent by default, so no drawing is needed.
  return canvas.toDataURL('image/png').split(',')[1];
}

/**
 * Normalizes an image file to a specific dimension by placing it centered on a
 * transparent canvas of the target size. This preserves the original image's
 * aspect ratio while standardizing its dimensions.
 * @param file The image file to process.
 * @param targetWidth The width of the output canvas.
 * @param targetHeight The height of the output canvas.
 * @returns A promise that resolves to the base64 encoded string of the normalized image.
 */
function normalizeImageToBase64(
  file: File,
  targetWidth: number,
  targetHeight: number,
): Promise<{ data: string; mimeType: string }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = document.createElement('canvas');
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return reject(new Error('Could not get 2D canvas context.'));
      }

      const imgAspectRatio = img.naturalWidth / img.naturalHeight;
      const canvasAspectRatio = targetWidth / targetHeight;

      let drawWidth = targetWidth;
      let drawHeight = targetHeight;

      if (imgAspectRatio > canvasAspectRatio) {
        // Image is wider than canvas, so width is the constraint
        drawWidth = targetWidth;
        drawHeight = targetWidth / imgAspectRatio;
      } else {
        // Image is taller than or equal to canvas aspect ratio, so height is the constraint
        drawHeight = targetHeight;
        drawWidth = targetHeight * imgAspectRatio;
      }

      const x = (targetWidth - drawWidth) / 2;
      const y = (targetHeight - drawHeight) / 2;

      // Draw the image centered on the new canvas
      ctx.drawImage(img, x, y, drawWidth, drawHeight);

      // We always output PNG to support transparency from the padding.
      const dataUrl = canvas.toDataURL('image/png');
      const base64Data = dataUrl.split(',')[1];

      resolve({ data: base64Data, mimeType: 'image/png' });
    };

    img.onerror = (err) => {
      URL.revokeObjectURL(url);
      reject(err);
    };

    img.src = url;
  });
}

/**
 * Gets all files for a given input, combining local and Drive files.
 */
function getFilesForInput(inputId: string): File[] {
  const inputEl = document.getElementById(inputId) as HTMLInputElement;
  const localFiles = inputEl.files ? Array.from(inputEl.files) : [];
  const driveSelectedFiles = driveFiles.get(inputId) || [];
  return [...localFiles, ...driveSelectedFiles];
}

// --- Canvas & Layer Logic ---

/**
 * Updates the enabled/disabled state of the layer action buttons.
 */
function updateLayerPanelActions() {
  const isLayerSelected = activeImageId !== null;
  duplicateLayerButton.disabled = !isLayerSelected;
  deleteLayerButton.disabled = !isLayerSelected;
}

/**
 * Renders the list of layers in the layer panel UI.
 */
function renderLayerList() {
  layerList.innerHTML = ''; // Clear existing list

  // Render from top to bottom, so reverse the canvasImages array for display
  [...canvasImages]
    .reverse()
    .forEach((image) => {
      const li = document.createElement('li');
      li.className = 'layer-item';
      li.draggable = true;
      li.dataset.id = image.id.toString();

      if (image.id === activeImageId) li.classList.add('active');
      if (!image.isVisible) li.classList.add('hidden-layer');
      if (image.isLocked) li.classList.add('locked-layer');

      const thumbnail = document.createElement('img');
      thumbnail.src = image.img.src;
      thumbnail.className = 'layer-thumbnail';

      const nameContainer = document.createElement('div');
      nameContainer.className = 'layer-name-container';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'layer-name';
      nameSpan.textContent = image.name;

      const nameInput = document.createElement('input');
      nameInput.type = 'text';
      nameInput.className = 'layer-name-input';
      nameInput.value = image.name;
      nameInput.classList.add('hidden');

      nameContainer.appendChild(nameSpan);
      nameContainer.appendChild(nameInput);

      nameSpan.addEventListener('dblclick', () => {
        if (image.isLocked) return;
        nameSpan.classList.add('hidden');
        nameInput.classList.remove('hidden');
        nameInput.focus();
        nameInput.select();
      });

      const stopEditing = () => {
        const newName = nameInput.value.trim();
        if (newName) {
          image.name = newName;
          nameSpan.textContent = newName;
        }
        nameSpan.classList.remove('hidden');
        nameInput.classList.add('hidden');
      };

      nameInput.addEventListener('blur', stopEditing);
      nameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') stopEditing();
        if (e.key === 'Escape') {
          nameInput.value = image.name;
          stopEditing();
        }
      });

      const controls = document.createElement('div');
      controls.className = 'layer-controls';

      const visibilityBtn = document.createElement('button');
      visibilityBtn.setAttribute(
        'aria-label',
        image.isVisible ? 'Hide layer' : 'Show layer',
      );
      visibilityBtn.innerHTML = image.isVisible
        ? `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M10.5 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/><path d="M0 8s3-5.5 8-5.5S16 8 16 8s-3 5.5-8 5.5S0 8 0 8zm8 3.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/></svg>`
        : `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="m10.79 12.912-1.614-1.615a3.5 3.5 0 0 1-4.474-4.474l-2.06-2.06C.938 6.278 0 8 0 8s3 5.5 8 5.5a7.029 7.029 0 0 0 2.79-.588zM5.21 3.088A7.028 7.028 0 0 1 8 2.5c5 0 8 5.5 8 5.5s-.939 1.721-2.641 3.238l-2.062-2.062a3.5 3.5 0 0 0-4.474-4.474L5.21 3.089z"/><path d="M5.525 7.646a2.5 2.5 0 0 0 2.829 2.829l-2.83-2.829zm4.95.708-2.829-2.83a2.5 2.5 0 0 1 2.829 2.829zm3.171 6-12-12 .708-.708 12 12-.708.708z"/></svg>`;
      visibilityBtn.onclick = (e) => {
        e.stopPropagation();
        image.isVisible = !image.isVisible;
        drawCanvas();
        renderLayerList();
      };

      const lockBtn = document.createElement('button');
      lockBtn.setAttribute(
        'aria-label',
        image.isLocked ? 'Unlock layer' : 'Lock layer',
      );
      lockBtn.innerHTML = image.isLocked
        ? `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M11 1a2 2 0 0 0-2 2v4a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h5V3a3 3 0 0 1 6 0v4a.5.5 0 0 1-1 0V3a2 2 0 0 0-2-2zM3 8a1 1 0 0 0-1 1v5a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V9a1 1 0 0 0-1-1H3z"/></svg>`
        : `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 1a2 2 0 0 1 2 2v4H6V3a2 2 0 0 1 2-2zm3 6V3a3 3 0 0 0-6 0v4a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2zM5 8h6a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/></svg>`;
      lockBtn.onclick = (e) => {
        e.stopPropagation();
        image.isLocked = !image.isLocked;
        renderLayerList();
        drawCanvas();
      };

      controls.appendChild(visibilityBtn);
      controls.appendChild(lockBtn);

      li.appendChild(thumbnail);
      li.appendChild(nameContainer);
      li.appendChild(controls);

      li.addEventListener('click', () => {
        activeImageId = image.id;
        drawCanvas();
        renderLayerList();
        updateLayerPanelActions();
      });

      // Drag and Drop
      li.addEventListener('dragstart', (e) => {
        e.dataTransfer!.setData('text/plain', image.id.toString());
        li.classList.add('dragging');
      });
      li.addEventListener('dragend', () => li.classList.remove('dragging'));

      layerList.appendChild(li);
    });
}

function duplicateLayer(id: number) {
  const imageToDuplicate = canvasImages.find((img) => img.id === id);
  if (!imageToDuplicate) return;

  const newImage: CanvasImage = {
    ...imageToDuplicate,
    id: Date.now(),
    x: imageToDuplicate.x + 20,
    y: imageToDuplicate.y + 20,
    name: `${imageToDuplicate.name} copy`,
  };
  const originalIndex = canvasImages.findIndex((img) => img.id === id);
  canvasImages.splice(originalIndex + 1, 0, newImage);
  activeImageId = newImage.id;
  drawCanvas();
  renderLayerList();
  updateLayerPanelActions();
}

function deleteLayer(id: number) {
  canvasImages = canvasImages.filter((img) => img.id !== id);
  if (activeImageId === id) {
    activeImageId = null;
  }
  drawCanvas();
  renderLayerList();
  updateUI();
  updateLayerPanelActions();
}

layerList.addEventListener('dragover', (e) => e.preventDefault());
layerList.addEventListener('drop', (e) => {
  e.preventDefault();
  const draggedId = parseInt(e.dataTransfer!.getData('text/plain'), 10);
  const targetElement = (e.target as HTMLElement).closest<HTMLElement>(
    '.layer-item',
  );
  if (!targetElement) return;

  const targetId = parseInt(targetElement.dataset.id!, 10);
  if (draggedId === targetId) return;

  const draggedImage = canvasImages.find((img) => img.id === draggedId);
  if (!draggedImage) return;

  const items = canvasImages.filter((img) => img.id !== draggedId);
  const targetIndex = items.findIndex((img) => img.id === targetId);

  items.splice(targetIndex + 1, 0, draggedImage);

  canvasImages = items;
  drawCanvas();
  renderLayerList();
});

/**
 * Gets the logical canvas coordinates of the resize handles' centers for a given image.
 */
function getHandlePositions(image: CanvasImage) {
  return {
    nw: { x: image.x, y: image.y },
    ne: { x: image.x + image.width, y: image.y },
    sw: { x: image.x, y: image.y + image.height },
    se: { x: image.x + image.width, y: image.y + image.height },
  };
}

/**
 * Checks if a given logical coordinate is over a resize handle of a specific image.
 */
function getHandleAtPos(
  image: CanvasImage,
  x: number,
  y: number,
): Handle | null {
  const handles = getHandlePositions(image);
  const handleHitboxSize = HANDLE_SIZE * 1.5;

  for (const key in handles) {
    const handle = handles[key as keyof typeof handles];
    if (
      x >= handle.x - handleHitboxSize / 2 &&
      x <= handle.x + handleHitboxSize / 2 &&
      y >= handle.y - handleHitboxSize / 2 &&
      y <= handle.y + handleHitboxSize / 2
    ) {
      return key as Handle;
    }
  }
  return null;
}

/**
 * Draws all images and resize handles for the active image onto the canvas.
 */
function drawCanvas() {
  if (!canvasCtx) return;
  canvasCtx.clearRect(0, 0, outpaintCanvas.width, outpaintCanvas.height);

  for (const image of canvasImages) {
    if (image.isVisible) {
      canvasCtx.drawImage(
        image.img,
        image.x,
        image.y,
        image.width,
        image.height,
      );
    }
  }

  const activeImage = canvasImages.find((img) => img.id === activeImageId);
  if (activeImage && activeImage.isVisible && !activeImage.isLocked) {
    canvasCtx.fillStyle = HANDLE_COLOR;
    canvasCtx.strokeStyle = HANDLE_STROKE;
    canvasCtx.lineWidth = 3;
    const handles = getHandlePositions(activeImage);
    for (const key in handles) {
      const pos = handles[key as keyof typeof handles];
      canvasCtx.beginPath();
      canvasCtx.arc(pos.x, pos.y, HANDLE_SIZE / 2, 0, 2 * Math.PI);
      canvasCtx.fill();
      canvasCtx.stroke();
    }
  }
}

/**
 * Updates the canvas size label visibility and content.
 */
function updateCanvasSizeLabel() {
  if (outpaintCanvas.width > 0 && outpaintCanvas.height > 0) {
    canvasSizeLabel.textContent = `${outpaintCanvas.width} x ${outpaintCanvas.height}`;
    canvasSizeLabel.classList.remove('hidden');
  } else {
    canvasSizeLabel.classList.add('hidden');
  }
}

/**
 * Sets the canvas logical and display dimensions based on the aspect ratio dropdown.
 */
function setupCanvasDimensions() {
  const selectedOptionText =
    aspectRatioSelect.options[aspectRatioSelect.selectedIndex].text;
  const resolutionMatch = selectedOptionText.match(/(\d+)x(\d+)/);

  if (resolutionMatch) {
    outpaintCanvas.width = parseInt(resolutionMatch[1], 10);
    outpaintCanvas.height = parseInt(resolutionMatch[2], 10);
  } else {
    // Fallback based on ratio string, in case the text format changes.
    const aspectRatio = aspectRatioSelect.value;
    switch (aspectRatio) {
      case '16:9':
        outpaintCanvas.width = 1344;
        outpaintCanvas.height = 738;
        break;
      case '9:16':
        outpaintCanvas.width = 738;
        outpaintCanvas.height = 1344;
        break;
      case '4:3':
        outpaintCanvas.width = 1024;
        outpaintCanvas.height = 768;
        break;
      case '3:4':
        outpaintCanvas.width = 768;
        outpaintCanvas.height = 1024;
        break;
      case '1:1':
      default:
        outpaintCanvas.width = 1024;
        outpaintCanvas.height = 1024;
        break;
    }
  }

  // Display size is handled by CSS (max-width/max-height on the canvas element)
  drawCanvas();
  updateCanvasSizeLabel();
}

/**
 * Adds a collection of files as images to the canvas asynchronously.
 */
async function addImagesToCanvas(files: File[]) {
  if (files.length === 0) return;

  if (outpaintCanvas.width === 0 || outpaintCanvas.height === 0) {
    setupCanvasDimensions();
  }

  const imageLoadPromises = files.map((file, index) => {
    return new Promise<void>((resolve, reject) => {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.onload = () => {
        URL.revokeObjectURL(url);

        let width = img.naturalWidth;
        let height = img.naturalHeight;
        const maxWidth = outpaintCanvas.width;
        const maxHeight = outpaintCanvas.height;
        const imgAspectRatio = width / height;

        // If image is larger than canvas, scale it down to fit while maintaining aspect ratio.
        if (width > maxWidth || height > maxHeight) {
          if (maxWidth / maxHeight > imgAspectRatio) {
            // Canvas is wider than image, so height is the limiting factor
            height = maxHeight;
            width = height * imgAspectRatio;
          } else {
            // Canvas is taller than/equal aspect to image, so width is the limiting factor
            width = maxWidth;
            height = width / imgAspectRatio;
          }
        }

        const x = (outpaintCanvas.width - width) / 2 + index * 20;
        const y = (outpaintCanvas.height - height) / 2 + index * 20;

        const newImage: CanvasImage = {
          id: Date.now() + index,
          name: file.name,
          img,
          x,
          y,
          width,
          height,
          isVisible: true,
          isLocked: false,
        };

        canvasImages.push(newImage);
        activeImageId = newImage.id;
        resolve();
      };
      img.onerror = reject;
      img.src = url;
    });
  });

  await Promise.all(imageLoadPromises);

  drawCanvas();
  renderLayerList();
  updateUI();
  updateLayerPanelActions();
}

/**
 * Gets mouse/touch position in logical canvas coordinates.
 */
function getMousePos(e: MouseEvent) {
  const rect = outpaintCanvas.getBoundingClientRect();
  const scaleX = outpaintCanvas.width / rect.width;
  const scaleY = outpaintCanvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

function getTouchPos(e: TouchEvent) {
  const rect = outpaintCanvas.getBoundingClientRect();
  const scaleX = outpaintCanvas.width / rect.width;
  const scaleY = outpaintCanvas.height / rect.height;
  const touch = e.touches[0];
  return {
    x: (touch.clientX - rect.left) * scaleX,
    y: (touch.clientY - rect.top) * scaleY,
  };
}

/**
 * Handles the start of a drag or resize operation from any input source.
 */
function handleInteractionStart(pos: { x: number; y: number }) {
  let clickedOnSomething = false;

  const activeImage = canvasImages.find((img) => img.id === activeImageId);
  if (activeImage && activeImage.isVisible && !activeImage.isLocked) {
    activeHandle = getHandleAtPos(activeImage, pos.x, pos.y);
    if (activeHandle) {
      isResizing = true;
      clickedOnSomething = true;
    }
  }

  if (!clickedOnSomething) {
    for (let i = canvasImages.length - 1; i >= 0; i--) {
      const image = canvasImages[i];
      if (
        image.isVisible &&
        pos.x >= image.x &&
        pos.x <= image.x + image.width &&
        pos.y >= image.y &&
        pos.y <= image.y + image.height
      ) {
        if (!image.isLocked) {
          isDragging = true;
          dragStart.x = pos.x - image.x;
          dragStart.y = pos.y - image.y;
          const [clickedImage] = canvasImages.splice(i, 1);
          canvasImages.push(clickedImage);
        }
        activeImageId = image.id;
        clickedOnSomething = true;
        break;
      }
    }
  }

  if (!clickedOnSomething) {
    activeImageId = null;
  }

  drawCanvas();
  renderLayerList();
  updateLayerPanelActions();
}

/**
 * Handles movement during a drag or resize operation.
 */
function handleInteractionMove(pos: { x: number; y: number }) {
  const activeImage = canvasImages.find((img) => img.id === activeImageId);
  if (!activeImage || activeImage.isLocked) return;

  if (isResizing && activeHandle) {
    const aspectRatio =
      activeImage.img.naturalWidth / activeImage.img.naturalHeight;
    let newWidth = activeImage.width;
    let newHeight = activeImage.height;
    const originalRight = activeImage.x + activeImage.width;
    const originalBottom = activeImage.y + activeImage.height;
    let widthChanged = false;

    if (activeHandle.includes('e')) {
      newWidth = pos.x - activeImage.x;
      widthChanged = true;
    } else if (activeHandle.includes('w')) {
      newWidth = originalRight - pos.x;
      widthChanged = true;
    }
    if (activeHandle.includes('s')) {
      newHeight = pos.y - activeImage.y;
      if (!widthChanged) newWidth = newHeight * aspectRatio;
    } else if (activeHandle.includes('n')) {
      newHeight = originalBottom - pos.y;
      if (!widthChanged) newWidth = newHeight * aspectRatio;
    }
    if (widthChanged) {
      newHeight = newWidth / aspectRatio;
    }
    if (activeHandle.includes('n')) {
      activeImage.y = originalBottom - newHeight;
    }
    if (activeHandle.includes('w')) {
      activeImage.x = originalRight - newWidth;
    }

    if (newWidth > 20 && newHeight > 20) {
      activeImage.width = newWidth;
      activeImage.height = newHeight;
    }
  } else if (isDragging) {
    activeImage.x = pos.x - dragStart.x;
    activeImage.y = pos.y - dragStart.y;
  }

  drawCanvas();
}

/**
 * Handles the end of a drag or resize operation.
 */
function handleInteractionEnd() {
  isDragging = false;
  isResizing = false;
  activeHandle = null;
  outpaintCanvas.style.cursor = 'default';
}

// --- Mouse and Touch Event Wrappers ---
function handleCanvasMouseDown(e: MouseEvent) {
  e.preventDefault();
  handleInteractionStart(getMousePos(e));
}

function handleCanvasMouseMove(e: MouseEvent) {
  e.preventDefault();
  const pos = getMousePos(e);
  if (isDragging || isResizing) {
    handleInteractionMove(pos);
  }
  // Update cursor (mouse-only feature)
  let cursor = 'default';
  const activeImage = canvasImages.find((img) => img.id === activeImageId);

  if (isDragging) {
    cursor = 'grabbing';
  } else if (activeImage?.isLocked) {
    cursor = 'not-allowed';
  } else {
    if (activeImage && activeImage.isVisible) {
      const handle = getHandleAtPos(activeImage, pos.x, pos.y);
      if (handle) {
        cursor =
          handle === 'nw' || handle === 'se' ? 'nwse-resize' : 'nesw-resize';
      }
    }
    if (cursor === 'default') {
      for (let i = canvasImages.length - 1; i >= 0; i--) {
        const image = canvasImages[i];
        if (
          image.isVisible &&
          pos.x >= image.x &&
          pos.x <= image.x + image.width &&
          pos.y >= image.y &&
          pos.y <= image.y + image.height
        ) {
          cursor = image.isLocked ? 'not-allowed' : 'grab';
          break;
        }
      }
    }
  }
  outpaintCanvas.style.cursor = cursor;
}

function handleCanvasMouseUp() {
  handleInteractionEnd();
}

function handleCanvasTouchStart(e: TouchEvent) {
  if (e.touches.length > 1) return;
  e.preventDefault();
  handleInteractionStart(getTouchPos(e));
}

function handleCanvasTouchMove(e: TouchEvent) {
  if (e.touches.length > 1) return;
  e.preventDefault();
  if (isDragging || isResizing) {
    handleInteractionMove(getTouchPos(e));
  }
}

function handleCanvasTouchEnd() {
  handleInteractionEnd();
}

// --- Google Drive Picker Logic ---
class GoogleDrivePicker {
  private tokenClient: any;
  private accessToken: string | null = null;
  private gapiLoaded = false;
  private gisLoaded = false;

  constructor() {
    this.loadGis();
    this.loadGapi();
  }

  private loadGapi() {
    const script = document.createElement('script');
    script.src = 'https://apis.google.com/js/api.js';
    script.async = true;
    script.defer = true;
    script.onload = () => {
      gapi.load('client:picker', () => {
        this.gapiLoaded = true;
      });
    };
    document.body.appendChild(script);
  }

  private loadGis() {
    const script = document.createElement('script');
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.defer = true;
    script.onload = () => {
      this.tokenClient = google.accounts.oauth2.initTokenClient({
        client_id: GOOGLE_CLIENT_ID,
        scope: DRIVE_SCOPES,
        callback: (tokenResponse: any) => {
          this.accessToken = tokenResponse.access_token;
        },
      });
      this.gisLoaded = true;
    };
    document.body.appendChild(script);
  }

  async showPicker(targetInputId: string, allowMultiple: boolean) {
    if (!GOOGLE_CLIENT_ID) {
      alert('Google Client ID is not configured.');
      return;
    }
    if (!this.gapiLoaded || !this.gisLoaded) {
      alert('Google API scripts are not loaded yet. Please try again.');
      return;
    }

    const show = () => {
      const view = new google.picker.View(google.picker.ViewId.DOCS);
      view.setMimeTypes('image/png,image/jpeg,image/jpg,image/webp');
      const picker = new google.picker.PickerBuilder()
        .enableFeature(google.picker.Feature.MULTISELECT_ENABLED)
        .enableFeature(google.picker.Feature.NAV_HIDDEN)
        .setAppId(API_KEY)
        .setOAuthToken(this.accessToken)
        .addView(view)
        .setDeveloperKey(API_KEY)
        .setCallback((data: any) =>
          this.pickerCallback(data, targetInputId, allowMultiple),
        )
        .build();
      picker.setVisible(true);
    };

    if (!this.accessToken) {
      this.tokenClient.callback = (tokenResponse: any) => {
        this.accessToken = tokenResponse.access_token;
        show();
      };
      this.tokenClient.requestAccessToken({ prompt: 'consent' });
    } else {
      show();
    }
  }

  private async pickerCallback(
    data: any,
    targetInputId: string,
    allowMultiple: boolean,
  ) {
    if (data.action === google.picker.Action.PICKED) {
      const files: File[] = [];
      for (const doc of data.docs) {
        const res = await fetch(
          `https://www.googleapis.com/drive/v3/files/${doc.id}?alt=media`,
          {
            headers: { Authorization: `Bearer ${this.accessToken}` },
          },
        );
        const blob = await res.blob();
        const file = new File([blob], doc.name, { type: doc.mimeType });
        files.push(file);
      }

      if (targetInputId === 'imageFile') {
        await addImagesToCanvas(files);
      } else {
        if (allowMultiple) {
          const existingFiles = driveFiles.get(targetInputId) || [];
          driveFiles.set(targetInputId, [...existingFiles, ...files]);
        } else {
          driveFiles.set(targetInputId, files);
        }
        updateImagePreviews(targetInputId);
      }
      updateUI();
    }
  }
}

const picker = new GoogleDrivePicker();

// --- Core Application Logic ---
/**
 * Processes and displays the response from the Gemini model.
 */
function processGeminiResponse(response: GenerateContentResponse) {
  let imageFound = false;
  let textFound = false;
  // Clear previous outputs before processing new ones.
  outputImage.src = '';
  textOutput.textContent = '';
  imageOutputContainer.classList.add('hidden');
  textOutputContainer.classList.add('hidden');
  lastGeneratedImageDataUrl = null;

  if (!response.candidates || response.candidates.length === 0) {
    textOutput.textContent = "The model didn't return a valid response.";
    textOutputContainer.classList.remove('hidden');
    textFound = true; // Still counts as a text response for UI purposes.
    return;
  }

  for (const part of response.candidates[0].content.parts) {
    if (part.inlineData) {
      const base64ImageBytes: string = part.inlineData.data;
      lastGeneratedImageDataUrl = `data:${part.inlineData.mimeType};base64,${base64ImageBytes}`;
      outputImage.src = lastGeneratedImageDataUrl;
      imageOutputContainer.classList.remove('hidden');
      imageFound = true;
    } else if (part.text) {
      textOutput.textContent = part.text;
      textFound = true;
    }
  }

  // Logic to decide which output to show. Prioritize image.
  if (imageFound) {
    // If there's an image, we don't show the text output even if it exists.
    textOutputContainer.classList.add('hidden');
  } else if (textFound) {
    // Show text output only if no image was found.
    if (!textOutput.textContent?.trim()) {
      textOutput.textContent =
        "The model didn't return an image. Please try adjusting your prompt.";
    }
    textOutputContainer.classList.remove('hidden');
  } else {
    // Fallback if no content parts were found at all.
    textOutput.textContent =
      "The model returned an empty response. Please try again.";
    textOutputContainer.classList.remove('hidden');
  }
}

/**
 * Handles generating an image from text prompts using Imagen.
 */
async function generateImage() {
  // Clear previous output state
  imageOutputContainer.classList.add('hidden');
  textOutputContainer.classList.add('hidden');
  lastGeneratedImageDataUrl = null;

  const prompt = [
    globalPrompt.value,
    faceDetails.value,
    dressPose.value,
    background.value,
  ]
    .filter(Boolean)
    .join(', ');

  if (!prompt.trim()) {
    textOutput.textContent = 'Please enter some details for the image.';
    textOutputContainer.classList.remove('hidden');
    return;
  }

  const aspectRatio = aspectRatioSelect.value;

  const response = await ai.models.generateImages({
    model: 'imagen-4.0-generate-001',
    prompt: prompt,
    config: {
      numberOfImages: 1,
      outputMimeType: 'image/jpeg',
      aspectRatio: aspectRatio,
    },
  });

  const base64ImageBytes: string = response.generatedImages[0].image.imageBytes;
  lastGeneratedImageDataUrl = `data:image/jpeg;base64,${base64ImageBytes}`;
  outputImage.src = lastGeneratedImageDataUrl;
  imageOutputContainer.classList.remove('hidden');
}

/**
 * Collects files from all reference inputs.
 */
function getReferenceFiles(): File[] {
  return [
    ...getFilesForInput('boyFaceImageFile'),
    ...getFilesForInput('girlFaceImageFile'),
    ...getFilesForInput('dressPoseImageFile'),
    ...getFilesForInput('backgroundImageFile'),
  ];
}

/**
 * Handles generating an image from text and optional reference images using Gemini.
 * To ensure the output respects the selected aspect ratio, all reference images are
 * first normalized by being placed onto a canvas of the target dimensions. This
 * provides a consistent, unambiguous size hint to the model.
 */
async function generateWithGemini() {
  // Clear previous output state
  imageOutputContainer.classList.add('hidden');
  textOutputContainer.classList.add('hidden');
  lastGeneratedImageDataUrl = null;

  const referenceFiles = getReferenceFiles();
  const userPromptParts = [
    globalPrompt.value,
    faceDetails.value,
    dressPose.value,
    background.value,
  ];
  const userPrompt = userPromptParts.filter(Boolean).join(', ');

  if (!userPrompt.trim() && referenceFiles.length === 0) {
    textOutput.textContent =
      'Please enter a prompt or provide a reference image.';
    textOutputContainer.classList.remove('hidden');
    return;
  }

  const parts: (
    | { inlineData: { data: string; mimeType: string } }
    | { text: string }
  )[] = [];

  // 1. Determine target resolution.
  const aspectRatio = aspectRatioSelect.value;
  const selectedOptionText =
    aspectRatioSelect.options[aspectRatioSelect.selectedIndex].text;
  const resolutionMatch = selectedOptionText.match(/(\d+)x(\d+)/);

  let width = 1024;
  let height = 1024;

  if (resolutionMatch) {
    width = parseInt(resolutionMatch[1], 10);
    height = parseInt(resolutionMatch[2], 10);
  } else {
    // Fallback for safety, in case the text format changes.
    switch (aspectRatio) {
      case '16:9':
        width = 1344;
        height = 738;
        break;
      case '9:16':
        width = 738;
        height = 1344;
        break;
      case '4:3':
        width = 1024;
        height = 768;
        break;
      case '3:4':
        width = 768;
        height = 1024;
        break;
      case '1:1':
      default:
        width = 1024;
        height = 1024;
        break;
    }
  }

  // 2. If there are reference images, normalize them to the target size.
  // This removes dimensional ambiguity.
  if (referenceFiles.length > 0) {
    const imagePartsPromises = referenceFiles.map(async (file) => {
      const { data, mimeType } = await normalizeImageToBase64(
        file,
        width,
        height,
      );
      return {
        inlineData: {
          data,
          mimeType,
        },
      };
    });
    const imageParts = await Promise.all(imagePartsPromises);
    parts.push(...imageParts);
  } else {
    // 3. If there are no reference images, send a blank canvas to set the dimensions.
    const blankCanvasBase64 = await createBlankCanvasAsBase64(width, height);
    parts.push({
      inlineData: {
        data: blankCanvasBase64,
        mimeType: 'image/png',
      },
    });
  }

  // 4. Construct a text prompt that explicitly states the desired resolution.
  const finalPrompt = [
    userPrompt.trim() ||
      'A beautiful image based on the provided visual references.',
    `The final image must have a resolution of exactly ${width}x${height} pixels.`,
  ]
    .join(' ')
    .trim();
  parts.push({ text: finalPrompt });

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash-image',
    contents: { parts },
    config: {
      responseModalities: [Modality.IMAGE, Modality.TEXT],
    },
  });

  processGeminiResponse(response);
}

/**
 * Handles editing an image, including outpainting, based on canvas state and text prompts.
 */
async function editImage() {
  // Clear previous output state
  imageOutputContainer.classList.add('hidden');
  textOutputContainer.classList.add('hidden');
  lastGeneratedImageDataUrl = null;

  const visibleImages = canvasImages.filter((img) => img.isVisible);
  if (visibleImages.length === 0) {
    textOutput.textContent =
      'Please add and enable at least one visible image on the canvas for editing.';
    textOutputContainer.classList.remove('hidden');
    return;
  }

  // Create a temporary canvas to composite only visible layers
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = outpaintCanvas.width;
  tempCanvas.height = outpaintCanvas.height;
  const tempCtx = tempCanvas.getContext('2d');

  if (tempCtx) {
    for (const image of canvasImages) {
      if (image.isVisible) {
        tempCtx.drawImage(
          image.img,
          image.x,
          image.y,
          image.width,
          image.height,
        );
      }
    }
  }

  const referenceFiles = getReferenceFiles();

  // The first "image part" is the content of the composited canvas.
  const canvasDataUrl = tempCanvas.toDataURL('image/png');
  const canvasBase64 = canvasDataUrl.split(',')[1];
  const canvasPart = {
    inlineData: {
      data: canvasBase64,
      mimeType: 'image/png',
    },
  };

  const allImageParts = [canvasPart];

  // Add other reference files
  const referencePartsPromises = referenceFiles.map(async (file) => {
    const data = await fileToBase64(file);
    return {
      inlineData: {
        data,
        mimeType: file.type,
      },
    };
  });
  const referenceParts = await Promise.all(referencePartsPromises);
  allImageParts.push(...referenceParts);

  const userPromptParts = [
    `Combine the visual elements on this canvas into a single, cohesive, and photorealistic scene with a final resolution of exactly ${outpaintCanvas.width}x${outpaintCanvas.height} pixels. Seamlessly blend the different objects and styles, and intelligently fill in all transparent areas to complete the picture.`,
    globalPrompt.value,
    faceDetails.value,
    dressPose.value,
    background.value,
  ];
  const finalPrompt = userPromptParts.filter(Boolean).join('. ');

  const textPart = { text: finalPrompt };

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash-image',
    contents: {
      parts: [...allImageParts, textPart],
    },
    config: {
      responseModalities: [Modality.IMAGE, Modality.TEXT],
    },
  });

  processGeminiResponse(response);
}

/**
 * Uses Gemini to generate a random, creative prompt.
 */
async function generateSurprisePrompt() {
  surpriseMeButton.disabled = true;
  surpriseMeButton.textContent = 'Generating...';
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents:
        'Generate a random, creative, and visually interesting image prompt for an AI image generator. The prompt should be a single, concise sentence.',
    });
    globalPrompt.value = response.text.trim();
    // Manually trigger input event for clear button visibility
    globalPrompt.dispatchEvent(new Event('input'));
  } catch (e: any) {
    console.error('Surprise prompt error', e);
    // Show a temporary error or just reset the button
  } finally {
    surpriseMeButton.disabled = false;
    surpriseMeButton.innerHTML = '✨ Surprise Me';
  }
}

// --- UI Event Handlers ---
async function handleGenerateClick() {
  setWorkspaceState('loading');

  const selectedModel = modelSelect.value;
  const hasCanvasImage = canvasImages.length > 0;

  try {
    if (selectedModel === 'imagen-4.0-generate-001') {
      await generateImage();
    } else {
      // Gemini model
      if (hasCanvasImage) {
        await editImage(); // Composition mode
      } else {
        await generateWithGemini(); // Generation mode
      }
    }
  } catch (e: any) {
    console.error('got error', e);
    textOutput.textContent = `An error occurred: ${e.message}`;
    textOutputContainer.classList.remove('hidden'); // Ensure error is shown
    imageOutputContainer.classList.add('hidden'); // Hide image container on error
  } finally {
    setWorkspaceState('output');
  }
}

/**
 * Manages the visibility of the main app sections: editor, loading, and output.
 */
function setWorkspaceState(state: WorkspaceState) {
  // Hide all major sections first
  editorWrapper.classList.add('hidden');
  loadingContainer.classList.add('hidden');
  outputWrapper.classList.add('hidden');
  generateButton.disabled = state === 'loading';
  clearCanvasButton.disabled = state === 'loading' || canvasImages.length === 0;

  if (state === 'editor') {
    editorWrapper.classList.remove('hidden');
    updateUI(); // This function will handle showing the correct view inside the editor
  } else if (state === 'loading') {
    loadingContainer.classList.remove('hidden');
  } else if (state === 'output') {
    outputWrapper.classList.remove('hidden');
    // The generate functions are responsible for showing the correct inner container.
    // This is a fallback check in case of an unhandled error.
    if (
      imageOutputContainer.classList.contains('hidden') &&
      textOutputContainer.classList.contains('hidden')
    ) {
      textOutput.textContent =
        textOutput.textContent ||
        'An unexpected error occurred. Please try again.';
      textOutputContainer.classList.remove('hidden');
    }
  }
}

/**
 * Updates the UI based on the selected model's capabilities and inputs.
 * This is primarily for the 'editor' state.
 */
function updateUI() {
  const selectedModel = modelSelect.value;
  const hasCanvasImage = canvasImages.length > 0;

  generateButton.disabled = false;

  if (selectedModel === 'imagen-4.0-generate-001') {
    generationOptionsContainer.classList.remove('hidden');
    imageEditUpload.classList.add('hidden');
    referenceImageSections.forEach((el) => {
      el.classList.add('hidden');
      el.open = false; // Close accordions
    });
    editorMainView.classList.add('hidden');
    imageFilePreview.classList.add('hidden');
    clearCanvasButton.classList.add('hidden');

    generateButton.textContent = 'Generate';
  } else {
    // Gemini model
    imageEditUpload.classList.remove('hidden');
    referenceImageSections.forEach((el) => el.classList.remove('hidden'));

    clearCanvasButton.disabled = !hasCanvasImage;
    clearCanvasButton.classList.toggle('hidden', !hasCanvasImage);

    // If there's a base image, show the canvas editor. Otherwise, show generation options.
    if (hasCanvasImage) {
      generationOptionsContainer.classList.remove('hidden'); // For setting canvas aspect ratio
      editorMainView.classList.remove('hidden');
      imageFilePreview.classList.add('hidden');
      generateButton.textContent = 'Compose & Generate';
    } else {
      generationOptionsContainer.classList.remove('hidden');
      editorMainView.classList.add('hidden');
      imageFilePreview.classList.remove('hidden'); // Show the grid preview for multi-upload
      generateButton.textContent = 'Generate';
    }
  }
}

// --- UI Setup Functions ---
/**
 * Sets up the clear buttons for all textareas.
 */
function setupClearButtons() {
  document.querySelectorAll('textarea').forEach((textarea) => {
    const wrapper = textarea.parentElement;
    if (!wrapper?.classList.contains('textarea-wrapper')) return;
    const clearButton = wrapper.querySelector(
      '.clear-button',
    ) as HTMLButtonElement;
    if (!clearButton) return;

    const toggle = () =>
      clearButton.classList.toggle('hidden', !textarea.value);
    clearButton.addEventListener('click', () => {
      textarea.value = '';
      toggle();
      textarea.focus();
    });
    textarea.addEventListener('input', toggle);
    toggle();
  });
}

/**
 * Updates the image preview for a given file input. For canvas images, it adds them to the canvas.
 */
async function updateImagePreviews(inputId: string) {
  const files = getFilesForInput(inputId);

  // Special handling for the main image file: init the canvas editor
  if (inputId === 'imageFile' && files.length > 0) {
    await addImagesToCanvas(files);
    // Clear the native file input after adding to canvas to allow re-selection of the same file
    imageFileInput.value = '';
    return;
  }

  // Standard handling for reference image previews
  const previewContainer = document.getElementById(`${inputId}Preview`);
  if (!previewContainer) return;

  const placeholder = previewContainer.querySelector('.preview-placeholder');

  // Clear only the images, leave the placeholder element alone
  previewContainer
    .querySelectorAll('.preview-image')
    .forEach((img) => img.remove());

  if (files.length > 0) {
    if (placeholder) placeholder.classList.add('hidden');
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = document.createElement('img');
        img.src = e.target?.result as string;
        img.alt = 'Image preview';
        img.classList.add('preview-image');
        previewContainer.appendChild(img);
      };
      reader.readAsDataURL(file);
    });
  } else {
    if (placeholder) placeholder.classList.remove('hidden');
  }
}

/**
 * Generates a timestamped filename.
 */
function generateFilename(prefix: string, extension: string): string {
  const date = new Date();
  const timestamp = `${date.getFullYear().toString()}${(date.getMonth() + 1)
    .toString()
    .padStart(2, '0')}${date.getDate().toString().padStart(2, '0')}-${date
    .getHours()
    .toString()
    .padStart(2, '0')}${date.getMinutes().toString().padStart(2, '0')}${date
    .getSeconds()
    .toString()
    .padStart(2, '0')}`;
  return `${prefix}-${timestamp}.${extension}`;
}

/**
 * Sets up the draggable splitter for resizing panels.
 */
function setupSplitter() {
  let isDragging = false;
  const isVertical = window.innerWidth <= 768; // Based on media query

  splitter.addEventListener('mousedown', (e) => {
    isDragging = true;
    document.body.style.cursor = isVertical ? 'row-resize' : 'col-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;

    if (isVertical) {
      // Vertical resizing (for mobile)
      const newHeight = window.innerHeight - e.clientY;
      const minHeight = 150;
      if (newHeight > minHeight) {
        layerPanel.style.height = `${newHeight}px`;
      }
    } else {
      // Horizontal resizing (for desktop)
      const newWidth = editorMainView.clientWidth - e.clientX;
      const minWidth = parseInt(getComputedStyle(layerPanel).minWidth, 10);
      const maxWidth = parseInt(getComputedStyle(layerPanel).maxWidth, 10);

      if (newWidth > minWidth && newWidth < maxWidth) {
        layerPanel.style.width = `${newWidth}px`;
      }
    }
  });

  document.addEventListener('mouseup', () => {
    isDragging = false;
    document.body.style.cursor = 'default';
    document.body.style.userSelect = 'auto';
  });
}

/**
 * Sets up event listeners and initial state.
 */
function main() {
  generateButton.addEventListener('click', handleGenerateClick);
  surpriseMeButton.addEventListener('click', generateSurprisePrompt);
  clearCanvasButton.addEventListener('click', () => {
    canvasImages = [];
    activeImageId = null;
    drawCanvas();
    renderLayerList();
    updateLayerPanelActions();
    updateUI();
  });
  modelSelect.addEventListener('change', updateUI);
  aspectRatioSelect.addEventListener('change', setupCanvasDimensions);

  // Setup file inputs and Drive buttons
  ALL_FILE_INPUT_IDS.forEach((id) => {
    const inputEl = document.getElementById(id) as HTMLInputElement;
    inputEl.addEventListener('change', async () => {
      await updateImagePreviews(id);
      updateUI();
    });
  });

  document.querySelectorAll('.drive-button').forEach((button) => {
    button.addEventListener('click', () => {
      const targetId = (button as HTMLElement).dataset.targetInput;
      if (targetId) {
        const targetInput = document.getElementById(
          targetId,
        ) as HTMLInputElement;
        picker.showPicker(targetId, targetInput.multiple);
      }
    });
  });

  // Canvas events for both mouse and touch
  outpaintCanvas.addEventListener('mousedown', handleCanvasMouseDown);
  document.addEventListener('mousemove', handleCanvasMouseMove);
  document.addEventListener('mouseup', handleCanvasMouseUp);
  outpaintCanvas.addEventListener('touchstart', handleCanvasTouchStart);
  document.addEventListener('touchmove', handleCanvasTouchMove);
  document.addEventListener('touchend', handleCanvasTouchEnd);

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    // Don't delete if user is typing in an input
    const activeEl = document.activeElement;
    if (activeEl && ['INPUT', 'TEXTAREA'].includes(activeEl.tagName)) return;

    if ((e.key === 'Delete' || e.key === 'Backspace') && activeImageId) {
      e.preventDefault();
      deleteLayer(activeImageId);
    }
  });

  // Output action buttons
  imageBackToEditorButton.addEventListener('click', () =>
    setWorkspaceState('editor'),
  );
  textBackToEditorButton.addEventListener('click', () =>
    setWorkspaceState('editor'),
  );

  addToCanvasButton.addEventListener('click', () => {
    if (!outputImage.src || outputImage.src.startsWith('blob:')) return;

    const img = new Image();
    img.crossOrigin = 'anonymous'; // Handle potential CORS issues if src was remote
    img.onload = () => {
      let width = img.naturalWidth;
      let height = img.naturalHeight;
      const maxWidth = outpaintCanvas.width;
      const maxHeight = outpaintCanvas.height;
      const imgAspectRatio = width / height;

      // If image is larger than canvas, scale it down to fit while maintaining aspect ratio.
      if (width > maxWidth || height > maxHeight) {
        if (maxWidth / maxHeight > imgAspectRatio) {
          height = maxHeight;
          width = height * imgAspectRatio;
        } else {
          width = maxWidth;
          height = width / imgAspectRatio;
        }
      }

      const newImage: CanvasImage = {
        id: Date.now(),
        name: 'Generated Image',
        img,
        x: (outpaintCanvas.width - width) / 2,
        y: (outpaintCanvas.height - height) / 2,
        width,
        height,
        isVisible: true,
        isLocked: false,
      };

      canvasImages.push(newImage);
      activeImageId = newImage.id;
      setWorkspaceState('editor');
      drawCanvas();
      renderLayerList();
      updateLayerPanelActions();
    };
    img.src = outputImage.src;
  });

  downloadImageButton.addEventListener('click', () => {
    if (!lastGeneratedImageDataUrl) return;

    const link = document.createElement('a');
    link.href = lastGeneratedImageDataUrl;

    // Extract mime type and extension from the data URL itself for a robust filename.
    const mimeTypeMatch = lastGeneratedImageDataUrl.match(
      /data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+).*,.*/,
    );
    let extension = 'png'; // Default extension
    if (mimeTypeMatch && mimeTypeMatch[1]) {
      extension = mimeTypeMatch[1].split('/')[1];
    }

    link.download = generateFilename('generated-image', extension);

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  downloadTextButton.addEventListener('click', () => {
    const text = textOutput.textContent;
    if (!text?.trim()) return;
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = generateFilename('generated-text', 'txt');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  });

  // Layer Panel Actions
  addLayerButton.addEventListener('click', () => imageFileInput.click());
  duplicateLayerButton.addEventListener('click', () => {
    if (activeImageId) duplicateLayer(activeImageId);
  });
  deleteLayerButton.addEventListener('click', () => {
    if (activeImageId) deleteLayer(activeImageId);
  });

  setupClearButtons();
  setupCanvasDimensions(); // Set initial canvas size
  setupSplitter();
  setWorkspaceState('editor'); // Set initial UI state
}

main();
