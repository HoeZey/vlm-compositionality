/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the CC-BY-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

var path = require("path");
var webpack = require("webpack");

module.exports = {
  entry: "./src/main.js",
  output: {
    path: __dirname,
    filename: "build/bundle.js",
  },
  resolve: {
    alias: {
      react: path.resolve("./node_modules/react"),
    },
    fallback: {
      net: false,
      dns: false,
    },
    extensions: ['.ts', '.js', '.tsx', '.jsx'],
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        loader: "babel-loader",
        exclude: /node_modules/,
        options: { presets: ["@babel/env"] },
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"],
      },
      {
        test: /\.(svg|png|jpe?g|ttf)$/,
        loader: "url-loader",
        options: { limit: 100000 },
      },
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.jpg$/,
        loader: "file-loader",
      },
    ],
  },
};
