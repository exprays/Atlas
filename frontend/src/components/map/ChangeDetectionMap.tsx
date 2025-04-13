/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useRef, useState } from 'react';
import mapboxgl, { GeoJSONSourceSpecification } from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

interface ChangeDetectionMapProps {
  geojsonUrl?: string;
  center?: [number, number];
  zoom?: number;
  width?: string;
  height?: string;
}

const ChangeDetectionMap: React.FC<ChangeDetectionMapProps> = ({
  geojsonUrl,
  center = [0, 0],
  zoom = 2,
  width = '100%',
  height = '500px',
}) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [geojsonData, setGeojsonData] = useState<any>(null);

  // Fetch GeoJSON data if URL provided
  useEffect(() => {
    if (!geojsonUrl) return;

    const fetchGeoJson = async () => {
      try {
        const response = await fetch(geojsonUrl);
        const data = await response.json();
        setGeojsonData(data);
      } catch (error) {
        console.error('Error fetching GeoJSON:', error);
      }
    };

    fetchGeoJson();
  }, [geojsonUrl]);

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current) return;

    mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || '';

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/satellite-v9',
      center,
      zoom
    });

    const mapObj = map.current;
    
    // Cleanup on unmount
    return () => {
      mapObj.remove();
    };
  }, []);

  // Add GeoJSON data to map when available
  useEffect(() => {
    if (!map.current || !geojsonData) return;
    
    const mapObj = map.current;

    // Wait for map to load
    if (!mapObj.loaded()) {
      mapObj.on('load', () => addGeoJsonToMap(mapObj, geojsonData));
    } else {
      addGeoJsonToMap(mapObj, geojsonData);
    }
  }, [geojsonData]);

  const addGeoJsonToMap = (mapObj: mapboxgl.Map, data: GeoJSONSourceSpecification['data']) => {
    // Remove existing layers and source if they exist
    if (mapObj.getLayer('changes-fill')) mapObj.removeLayer('changes-fill');
    if (mapObj.getLayer('changes-line')) mapObj.removeLayer('changes-line');
    if (mapObj.getSource('changes')) mapObj.removeSource('changes');

    // Add new source and layers
    mapObj.addSource('changes', {
      type: 'geojson',
      data
    });
    
    mapObj.addLayer({
      id: 'changes-fill',
      type: 'fill',
      source: 'changes',
      paint: {
        'fill-color': '#FF0000',
        'fill-opacity': 0.5
      }
    });
    
    mapObj.addLayer({
      id: 'changes-line',
      type: 'line',
      source: 'changes',
      paint: {
        'line-color': '#FF0000',
        'line-width': 2
      }
    });

    // Fit map to GeoJSON bounds if features exist
    if (data && typeof data === 'object' && 'features' in data && data.features.length > 0) {
      const bounds = new mapboxgl.LngLatBounds();
      data.features.forEach((feature: any) => {
        if (feature.geometry && feature.geometry.coordinates) {
          feature.geometry.coordinates.forEach((coord: any) => {
            if (Array.isArray(coord[0])) {
              // Polygon
              coord.forEach((point: number[]) => {
                bounds.extend([point[0], point[1]]);
              });
            } else {
              // Point
              bounds.extend([coord[0], coord[1]]);
            }
          });
        }
      });
      
      // Add padding and fit bounds
      mapObj.fitBounds(bounds, { padding: 50 });
    }
  };

  return (
    <div style={{ width, height }} className="relative rounded-lg overflow-hidden shadow-lg">
      <div ref={mapContainer} className="absolute top-0 left-0 w-full h-full" />
    </div>
  );
};

export default ChangeDetectionMap;