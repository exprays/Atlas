'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { getResults } from '@/lib/api';
import Link from 'next/link';
import ChangeDetectionResults from '@/components/analysis/ChangeDetectionResults';

export default function ResultsPage() {
  const { jobId } = useParams();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchResults = async () => {
      if (!jobId || typeof jobId !== 'string') {
        setError('Invalid job ID');
        setLoading(false);
        return;
      }

      try {
        const data = await getResults(jobId);
        setResults(data);
      } catch (err) {
        setError(`Failed to load results: ${err instanceof Error ? err.message : 'Unknown error'}`);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [jobId]);

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
                Upload New
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto mt-8 px-4 pb-16">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Change Detection Results</h1>
          <p className="text-gray-600">Job ID: {jobId}</p>
        </div>

        {loading ? (
          <div className="bg-white p-8 rounded-lg shadow flex flex-col items-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
            <p className="text-gray-700">Loading results...</p>
          </div>
        ) : error ? (
          <div className="bg-white p-8 rounded-lg shadow">
            <div className="p-4 bg-red-50 border border-red-200 rounded-md mb-4">
              <p className="text-red-600">{error}</p>
            </div>
            <Link 
              href="/upload"
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              Try Again
            </Link>
          </div>
        ) : results && results.status === 'completed' ? (
          <ChangeDetectionResults
            jobId={results.job_id}
            changePercentage={results.change_percentage}
            numRegions={results.num_regions}
            visualizationUrl={results.visualization_url}
            geojsonUrl={results.geojson_url}
            accuracy={results.accuracy}
            kappa={results.kappa}
            fiError={results.fi_error}
          />
        ) : (
          <div className="bg-white p-8 rounded-lg shadow">
            <p className="text-yellow-600 mb-4">Processing not complete. Status: {results?.status || 'unknown'}</p>
            <button 
              onClick={() => window.location.reload()}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              Refresh
            </button>
          </div>
        )}
      </main>
    </div>
  );
}