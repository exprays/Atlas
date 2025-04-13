import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Image from 'next/image';
import ChangeDetectionMap from '../map/ChangeDetectionMap';

interface ChangeDetectionResultsProps {
  jobId: string;
  changePercentage: number;
  numRegions: number;
  accuracy?: number;
  kappa?: number;
  fiError?: number;
  visualizationUrl: string;
  geojsonUrl: string;
}

const ChangeDetectionResults: React.FC<ChangeDetectionResultsProps> = ({
  jobId,
  changePercentage,
  numRegions,
  accuracy,
  kappa,
  fiError,
  visualizationUrl,
  geojsonUrl
}) => {
  // Prepare data for metrics chart
  const metricsData = [
    { name: 'Accuracy', value: accuracy || 0 },
    { name: 'Kappa', value: kappa || 0 },
    { name: 'FI Error', value: fiError || 0 }
  ].filter(metric => metric.value > 0);

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-4">Change Detection Results</h2>
      
      {/* Summary stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-800">Change Percentage</h3>
          <p className="text-3xl font-bold">{changePercentage.toFixed(2)}%</p>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-green-800">Change Regions</h3>
          <p className="text-3xl font-bold">{numRegions}</p>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-purple-800">Job ID</h3>
          <p className="text-xl font-medium">{jobId}</p>
        </div>
      </div>
      
      {/* Visualization and Map */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div>
          <h3 className="text-xl font-bold mb-3">Change Visualization</h3>
          <div className="relative h-[400px] w-full">
            <Image 
              src={visualizationUrl} 
              alt="Change Detection Visualization" 
              layout="fill"
              objectFit="contain"
              className="rounded-lg"
            />
          </div>
        </div>
        
        <div>
          <h3 className="text-xl font-bold mb-3">Change Map</h3>
          <ChangeDetectionMap 
            geojsonUrl={geojsonUrl}
            height="400px"
          />
        </div>
      </div>
      
      {/* Metrics Chart */}
      {metricsData.length > 0 && (
        <div>
          <h3 className="text-xl font-bold mb-3">Accuracy Metrics</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={metricsData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 1]} />
                <Tooltip formatter={(value) => (Number(value).toFixed(4))} />
                <Legend />
                <Bar dataKey="value" name="Value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChangeDetectionResults;