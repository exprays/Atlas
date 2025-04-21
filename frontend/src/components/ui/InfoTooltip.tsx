import { useState } from 'react';

interface InfoTooltipProps {
  text: string;
}

export default function InfoTooltip({ text }: InfoTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  
  return (
    <div className="inline-block relative ml-1">
      <button
        className="text-gray-500 h-5 w-5 rounded-full bg-gray-100 flex items-center justify-center text-xs focus:outline-none"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onClick={() => setIsVisible(!isVisible)}
      >
        i
      </button>
      {isVisible && (
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-48 p-2 bg-gray-900 text-white text-xs rounded shadow-lg z-10">
          {text}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 h-2 w-2 bg-gray-900 rotate-45"></div>
        </div>
      )}
    </div>
  );
}