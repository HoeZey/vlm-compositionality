#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

# Download images for HAKE Dataset
import os

import os
from urllib.request import urlretrieve
from tqdm import tqdm
import json as js
from pathlib import Path
import asyncio
import aiohttp
import aiofiles


async def download_image(session, url, to_folder, img):
    async with session.get(url) as response:
        async with aiofiles.open(os.path.join(to_folder, img), 'wb') as f:
            await f.write(await response.read())


async def image_url_download(url_file, to_folder):
    if not os.path.exists(to_folder):
        os.mkdir(to_folder)
    contents = js.load(open(Path.home().joinpath(url_file), 'r'))

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[
            download_image(session, url, to_folder, img) for img, url in tqdm(contents.items())
            if not os.path.exists(os.path.join(to_folder, img))
        ])


if __name__=='__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python download.py url_base imgname to_folder")
    else:
        asyncio.run(image_url_download(sys.argv[1], sys.argv[2]))