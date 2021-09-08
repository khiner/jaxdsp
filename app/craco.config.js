const path = require('path')
const fs = require('fs')
const cracoBabelLoader = require('craco-babel-loader')

// Handle relative paths to sibling packages (https://stackoverflow.com/a/58603207/780425)
const appDirectory = fs.realpathSync(process.cwd())
const resolvePackage = relativePath => path.resolve(appDirectory, relativePath)

module.exports = {
  webpack: {
    alias: {
      // Needed because using the local file dependency for ../client
      // will also grab react from its node_modules folder.
      react: path.resolve('./node_modules/react'),
    },
  },
  plugins: [
    {
      plugin: cracoBabelLoader,
      options: {
        includes: [
          // Fix "unexpected token" error importing ES6 directly:
          resolvePackage('../client'),
        ],
      },
    },
  ],
}
