import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Image from 'next/image';
import ChangeDetectionMap from '../map/ChangeDetectionMap';
import InfoTooltip from '../ui/InfoTooltip';

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

      {/* Accuracy Metrics Cards */}
      {(accuracy !== undefined || kappa !== undefined || fiError !== undefined) && (
        <div className="mb-6">
          <h3 className="text-xl font-bold mb-3">
            Accuracy Metrics
            <InfoTooltip text="These metrics evaluate how well the change detection algorithm performed compared to expected outcomes." />
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {accuracy !== undefined && (
              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-green-800">Overall Accuracy</h3>
                <p className="text-3xl font-bold">{(accuracy * 100).toFixed(2)}%</p>
                <p className="text-sm text-gray-600 mt-1">Percentage of correctly classified pixels</p>
              </div>
            )}
            
            {kappa !== undefined && (
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-blue-800">Kappa Coefficient</h3>
                <p className="text-3xl font-bold">{kappa.toFixed(4)}</p>
                <p className="text-sm text-gray-600 mt-1">Agreement between prediction and reality</p>
              </div>
            )}
            
            {fiError !== undefined && (
              <div className="bg-amber-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-amber-800">FI Error</h3>
                <p className="text-3xl font-bold">{(fiError * 100).toFixed(2)}%</p>
                <p className="text-sm text-gray-600 mt-1">False information proportion</p>
              </div>
            )}
          </div>
        </div>
      )}
      
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
        <div className="mt-8 bg-white p-6 rounded-lg shadow-md">
          <h3 className="text-xl font-bold mb-3">Accuracy Metrics Comparison</h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={metricsData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis 
                  domain={[0, 1]} 
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} 
                />
                <Tooltip 
                  formatter={(value) => [`${(Number(value) * 100).toFixed(2)}%`, 'Value']} 
                  labelFormatter={(label) => `Metric: ${label}`}
                />
                <Legend />
                <Bar 
                  dataKey="value" 
                  name="Value" 
                  fill="#8884d8" 
                  radius={[4, 4, 0, 0]}
                  label={{ 
                    position: 'top', 
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter: (value: any) => `${(Number(value) * 100).toFixed(1)}%` 
                  }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 text-sm text-gray-600">
            <p><span className="font-medium">Accuracy:</span> Percentage of pixels correctly classified (higher is better)</p>
            <p><span className="font-medium">Kappa:</span> Agreement between prediction and ground truth, accounting for chance (higher is better)</p>
            <p><span className="font-medium">FI Error:</span> Proportion of false positive changes (lower is better)</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChangeDetectionResults;