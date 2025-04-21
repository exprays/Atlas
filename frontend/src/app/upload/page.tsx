'use client';

import UploadForm from '@/components/upload/UploadForm';
import Link from 'next/link';

export default function UploadPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="container mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">AtlasEye</h1>
              <span className="ml-2 text-sm text-gray-600">Satellite Change Detection</span>
            </Link>
            <nav>
              <Link 
                href="/"
                className="text-gray-600 hover:text-gray-900 px-3 py-2"
              >
                Home
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto mt-8 px-4 pb-16">
        <div className="max-w-3xl mx-auto bg-white rounded-lg shadow p-8">
          <h1 className="text-3xl font-bold mb-6">Upload Images for Change Detection</h1>
          <p className="text-gray-600 mb-8">
            Upload satellite images captured before and after an event to detect and analyze changes.
            Supported formats: .tif, .jpg, .png
          </p>
          
          <UploadForm />
        </div>
      </main>
    </div>
  );
}