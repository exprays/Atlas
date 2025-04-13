import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white">
      {/* Hero Section */}
      <header className="container mx-auto px-6 py-16">
        <div className="flex flex-col md:flex-row items-center">
          <div className="md:w-1/2 mb-10 md:mb-0">
            <h1 className="text-4xl md:text-6xl font-bold mb-4">
              Detect Changes in Satellite Imagery
            </h1>
            <p className="text-xl text-gray-300 mb-8">
              AtlasEye uses advanced AI to identify and analyze changes between satellite images, providing insights for urban development, deforestation, disaster response, and more.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Link 
                href="/upload"
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors text-center"
              >
                Upload Images
              </Link>
              <Link 
                href="/demo"
                className="bg-transparent hover:bg-white hover:text-gray-800 text-white font-bold py-3 px-6 rounded-lg border border-white transition-colors text-center"
              >
                View Demo
              </Link>
            </div>
          </div>
          <div className="md:w-1/2">
            <div className="relative h-[300px] md:h-[400px] w-full">
              <Image 
                src="/satellite-comparison.png" 
                alt="Satellite imagery comparison"
                fill
                className="object-cover rounded-lg shadow-xl"
                priority
              />
            </div>
          </div>
        </div>
      </header>

      {/* Features */}
      <section className="bg-gray-800 py-16">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-gray-700 p-6 rounded-lg">
              <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <h3 className="text-xl font-bold mb-2">Advanced AI Detection</h3>
              <p className="text-gray-300">Powered by state-of-the-art deep learning models to detect even subtle changes with high accuracy.</p>
            </div>
            <div className="bg-gray-700 p-6 rounded-lg">
              <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                </svg>
              </div>
              <h3 className="text-xl font-bold mb-2">Geospatial Analysis</h3>
              <p className="text-gray-300">Export results as GeoJSON for integration with GIS systems and detailed spatial analysis.</p>
            </div>
            <div className="bg-gray-700 p-6 rounded-lg">
              <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-xl font-bold mb-2">Detailed Metrics</h3>
              <p className="text-gray-300">Get comprehensive analysis with accuracy metrics, change percentages, and region statistics.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-16">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">Applications</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="flex items-start">
              <div className="bg-blue-600 rounded-full p-2 mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-bold mb-2">Urban Development</h3>
                <p className="text-gray-300">Track construction, urban sprawl, and infrastructure changes over time.</p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="bg-blue-600 rounded-full p-2 mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-bold mb-2">Deforestation Monitoring</h3>
                <p className="text-gray-300">Identify and measure forest loss and vegetation changes.</p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="bg-blue-600 rounded-full p-2 mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-bold mb-2">Disaster Assessment</h3>
                <p className="text-gray-300">Assess damage after natural disasters for rapid response and recovery planning.</p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="bg-blue-600 rounded-full p-2 mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div>
                <h3 className="text-xl font-bold mb-2">Agricultural Monitoring</h3>
                <p className="text-gray-300">Monitor crop growth, harvest patterns, and land use changes.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="bg-blue-700 py-16">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Detect Changes?</h2>
          <p className="text-xl mb-8 max-w-2xl mx-auto">Upload your satellite images and get detailed analysis in minutes.</p>
          <Link 
            href="/upload"
            className="bg-white text-blue-700 hover:bg-gray-100 font-bold py-3 px-8 rounded-lg transition-colors inline-block"
          >
            Get Started
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 py-8">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-4 md:mb-0">
              <h3 className="text-xl font-bold">AtlasEye</h3>
              <p className="text-gray-400">Satellite Change Detection Platform</p>
            </div>
            <div className="flex space-x-4">
              <a href="/about" className="text-gray-400 hover:text-white">About</a>
              <a href="/contact" className="text-gray-400 hover:text-white">Contact</a>
              <a href="/privacy" className="text-gray-400 hover:text-white">Privacy</a>
            </div>
          </div>
          <div className="mt-8 text-center text-gray-500 text-sm">
            &copy; {new Date().getFullYear()} AtlasEye. All rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
