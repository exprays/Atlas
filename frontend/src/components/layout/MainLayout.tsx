import React from 'react';
import Link from 'next/link';

interface MainLayoutProps {
  children: React.ReactNode;
  title: string;
  subtitle?: string;
}

export default function MainLayout({ children, title, subtitle }: MainLayoutProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="container mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">AtlasEye</h1>
              <span className="ml-2 text-sm text-gray-600">Satellite Change Detection</span>
            </Link>
            <nav className="flex space-x-4">
              <Link 
                href="/"
                className="text-gray-600 hover:text-gray-900 px-3 py-2"
              >
                Home
              </Link>
              <Link 
                href="/upload"
                className="text-gray-600 hover:text-gray-900 px-3 py-2"
              >
                Upload
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto mt-8 px-4 pb-16">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">{title}</h1>
          {subtitle && <p className="text-gray-600">{subtitle}</p>}
        </div>
        {children}
      </main>
    </div>
  );
}