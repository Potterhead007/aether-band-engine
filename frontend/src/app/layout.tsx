import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from './providers'
import { ErrorBoundary } from '@/components/ErrorBoundary'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AETHER | AI Music Generation',
  description: 'Autonomous Ensemble for Thoughtful Harmonic Expression and Rendering',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ErrorBoundary>
        <Providers>
          <div className="min-h-screen flex flex-col">
            <nav className="border-b border-slate-700/50 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-aether-400 to-purple-500 flex items-center justify-center">
                      <span className="text-white font-bold text-sm">A</span>
                    </div>
                    <span className="text-xl font-bold gradient-text">AETHER</span>
                  </div>
                  <div className="flex items-center space-x-4">
                    <a href="/generate" className="text-slate-300 hover:text-white transition-colors">Generate</a>
                    <a href="/history" className="text-slate-300 hover:text-white transition-colors">History</a>
                    <a href="/docs" className="text-slate-300 hover:text-white transition-colors">API Docs</a>
                  </div>
                </div>
              </div>
            </nav>
            <main className="flex-1">
              {children}
            </main>
            <footer className="border-t border-slate-700/50 bg-slate-900/30 py-6">
              <div className="max-w-7xl mx-auto px-4 text-center text-slate-500 text-sm">
                AETHER Band Engine v0.1.0 - AI-Powered Music Generation
              </div>
            </footer>
          </div>
        </Providers>
        </ErrorBoundary>
      </body>
    </html>
  )
}
