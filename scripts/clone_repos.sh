cd ./benchmarks

git clone https://github.com/facebookresearch/DCI.git dci
git clone https://github.com/mertyg/vision-language-models-are-bows.git aro
git clone https://github.com/om-ai-lab/VL-CheckList.git dci/reproduction/densely_captioned_images/repro/eval/VLChecklist

# Remove unnecessary files
rm -f -r ./aro/.git aro/.gitignore ./aro/scripts ./aro/temp_data 
rm -f -r ./dci/.git
rm -f -r ./vlc/.git ./vl-checklist/.gitignore ./vl-checklist/example_models ./vl-checklist/docs

mv -f ./run_vlc.py ./dci/reproduction/densely_captioned_images/repro/eval/run_vlc.py
mv -f ./config.py ./dci/reproduction/densely_captioned_images/repro/eval/run_vlc.py
mv -f ./download.py ./dci/reproduction/densely_captioned_images/repro/setup_data/download.py
mv 