/**
 * API client functions for interacting with the AtlasEye backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

/**
 * Upload satellite images for change detection
 */
export async function uploadImages(beforeImage: File, afterImage: File, jobName?: string) {
  const formData = new FormData();
  formData.append('before_image', beforeImage);
  formData.append('after_image', afterImage);
  
  if (jobName) {
    formData.append('job_name', jobName);
  }

  const response = await fetch(`${API_BASE_URL}/detection/upload-images/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Process previously uploaded images
 */
export async function processImages(jobId: string) {
  const response = await fetch(`${API_BASE_URL}/detection/process/${jobId}`, {
    method: 'POST',
  });

  if (!response.ok) {
    throw new Error(`Processing failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get processing results
 */
export async function getResults(jobId: string) {
  const response = await fetch(`${API_BASE_URL}/detection/results/${jobId}`);

  if (!response.ok) {
    throw new Error(`Failed to get results: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Poll for results until processing is complete
 */
export async function pollForResults(jobId: string, maxAttempts = 30, interval = 2000) {
  let attempts = 0;
  
  // Check status repeatedly until complete or max attempts reached
  while (attempts < maxAttempts) {
    try {
      const results = await getResults(jobId);
      if (results.status === 'completed') {
        return results;
      }
    } catch (error) {
      // If results aren't ready yet, continue polling
      if (!(error instanceof Error) || !error.message.includes('404')) {
        throw error;
      }
    }
    
    // Wait before next attempt
    await new Promise(resolve => setTimeout(resolve, interval));
    attempts++;
  }
  
  throw new Error('Processing timed out');
}