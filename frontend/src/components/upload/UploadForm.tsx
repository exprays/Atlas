'use client';

import { useState, useRef, ChangeEvent, FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { uploadImages, processImages, pollForResults } from '@/lib/api';

// Interface for file upload state
interface FilePreview {
  file: File;
  previewUrl: string;
}

export default function UploadForm() {
  const router = useRouter();
  const [jobName, setJobName] = useState('');
  const [beforeImage, setBeforeImage] = useState<FilePreview | null>(null);
  const [afterImage, setAfterImage] = useState<FilePreview | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  const beforeInputRef = useRef<HTMLInputElement>(null);
  const afterInputRef = useRef<HTMLInputElement>(null);

  // Handle file selection for "before" image
  const handleBeforeImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onloadend = () => {
        setBeforeImage({
          file,
          previewUrl: reader.result as string,
        });
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle file selection for "after" image
  const handleAfterImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onloadend = () => {
        setAfterImage({
          file,
          previewUrl: reader.result as string,
        });
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle form submission
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    
    if (!beforeImage || !afterImage) {
      setError('Please select both before and after images');
      return;
    }

    try {
      // Step 1: Upload images
      setIsUploading(true);
      setUploadProgress(10);
      
      const uploadResult = await uploadImages(
        beforeImage.file,
        afterImage.file,
        jobName || undefined
      );
      
      setUploadProgress(50);
      
      // Step 2: Process images
      setIsUploading(false);
      setIsProcessing(true);
      
      const jobId = uploadResult.job_id;
      await processImages(jobId);
      
      // Step 3: Poll for results (optional - we can also just redirect)
      try {
        await pollForResults(jobId, 5); // Just try a few times quickly
      } catch (e) {
        // It's okay if polling times out, we'll redirect to results page anyway
      }
      
      // Step 4: Navigate to results page
      router.push(`/results/${jobId}`);
      
    } catch (err) {
      setIsUploading(false);
      setIsProcessing(false);
      setError(`Error: ${err instanceof Error ? err.message : 'An unknown error occurred'}`);
      console.error(err);
    }
  };

  // Clear selected file
  const clearImage = (type: 'before' | 'after') => {
    if (type === 'before') {
      setBeforeImage(null);
      if (beforeInputRef.current) beforeInputRef.current.value = '';
    } else {
      setAfterImage(null);
      if (afterInputRef.current) afterInputRef.current.value = '';
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* Job name input */}
      <div className="space-y-2">
        <label htmlFor="jobName" className="block text-sm font-medium">
          Job Name (optional)
        </label>
        <input
          type="text"
          id="jobName"
          value={jobName}
          onChange={(e) => setJobName(e.target.value)}
          placeholder="e.g. Urban Development Study"
          className="w-full px-4 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          disabled={isUploading || isProcessing}
        />
      </div>
      
      {/* Image upload section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Before Image */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <label className="block text-lg font-medium">Before Image</label>
            {beforeImage && (
              <button
                type="button"
                onClick={() => clearImage('before')}
                className="text-sm text-red-600 hover:text-red-800"
                disabled={isUploading || isProcessing}
              >
                Clear
              </button>
            )}
          </div>
          
          {beforeImage ? (
            <div className="relative h-64 bg-gray-100 rounded-lg overflow-hidden">
              <Image
                src={beforeImage.previewUrl}
                alt="Before image preview"
                fill
                className="object-contain"
              />
              <div className="absolute bottom-2 right-2 bg-white/80 px-2 py-1 rounded text-xs">
                {beforeImage.file.name}
              </div>
            </div>
          ) : (
            <div 
              onClick={() => beforeInputRef.current?.click()}
              className="h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center cursor-pointer hover:bg-gray-50"
            >
              <div className="text-center p-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mt-2 text-sm text-gray-600">Click to select Before image</p>
                <p className="text-xs text-gray-500 mt-1">.tif, .jpg, or .png</p>
              </div>
            </div>
          )}
          
          <input
            ref={beforeInputRef}
            type="file"
            accept=".tif,.tiff,.jpg,.jpeg,.png"
            onChange={handleBeforeImageChange}
            className="hidden"
            disabled={isUploading || isProcessing}
          />
        </div>
        
        {/* After Image */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <label className="block text-lg font-medium">After Image</label>
            {afterImage && (
              <button
                type="button"
                onClick={() => clearImage('after')}
                className="text-sm text-red-600 hover:text-red-800"
                disabled={isUploading || isProcessing}
              >
                Clear
              </button>
            )}
          </div>
          
          {afterImage ? (
            <div className="relative h-64 bg-gray-100 rounded-lg overflow-hidden">
              <Image
                src={afterImage.previewUrl}
                alt="After image preview"
                fill
                className="object-contain"
              />
              <div className="absolute bottom-2 right-2 bg-white/80 px-2 py-1 rounded text-xs">
                {afterImage.file.name}
              </div>
            </div>
          ) : (
            <div 
              onClick={() => afterInputRef.current?.click()}
              className="h-64 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center cursor-pointer hover:bg-gray-50"
            >
              <div className="text-center p-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mt-2 text-sm text-gray-600">Click to select After image</p>
                <p className="text-xs text-gray-500 mt-1">.tif, .jpg, or .png</p>
              </div>
            </div>
          )}
          
          <input
            ref={afterInputRef}
            type="file"
            accept=".tif,.tiff,.jpg,.jpeg,.png"
            onChange={handleAfterImageChange}
            className="hidden"
            disabled={isUploading || isProcessing}
          />
        </div>
      </div>
      
      {/* Progress and errors */}
      {isUploading && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Uploading images...</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${uploadProgress}%` }}></div>
          </div>
        </div>
      )}
      
      {isProcessing && (
        <div className="flex items-center space-x-2">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
          <p className="text-sm font-medium">Processing images. This may take a moment...</p>
        </div>
      )}
      
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}
      
      {/* Submit button */}
      <button
        type="submit"
        className="w-full py-3 px-6 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium rounded-lg transition-colors"
        disabled={!beforeImage || !afterImage || isUploading || isProcessing}
      >
        {isUploading ? 'Uploading...' : isProcessing ? 'Processing...' : 'Detect Changes'}
      </button>
    </form>
  );
}