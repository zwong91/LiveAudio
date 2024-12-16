/** @type {import('next').NextConfig} */
const fs = require('fs');
const path = require('path');

module.exports = {
  devServer: {
    https: {
      key: fs.readFileSync(path.join(__dirname, 'cf.key')),
      cert: fs.readFileSync(path.join(__dirname, 'cf.pem1')),
    },
  },
};
