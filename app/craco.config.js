const path = require('path');

module.exports = {
  webpack: {
    alias: {
      // Needed because using the local file dependency for ../client
      // will also grap react from its node_modules folder.
      react: path.resolve("./node_modules/react"),
    },
  },
};
