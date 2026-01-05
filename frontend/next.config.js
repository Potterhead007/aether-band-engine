const path = require('path')

/** @type {import('next').NextConfig} */
const nextConfig = {
  // output: 'standalone', // Only for self-hosted Docker deployments
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  webpack: (config) => {
    config.resolve.alias['@'] = path.join(__dirname, 'src')
    return config
  },
}

module.exports = nextConfig
