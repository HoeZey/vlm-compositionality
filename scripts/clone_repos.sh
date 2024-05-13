cd ./benchmarks

git clone https://github.com/mertyg/vision-language-models-are-bows.git aro
git clone https://github.com/om-ai-lab/VL-CheckList.git vlc
git clone https://github.com/facebookresearch/DCI.git dci

# Remove unnecessary files
rm -f -r ./aro/.git aro/.gitignore ./aro/scripts ./aro/temp_data 
rm -f -r ./dci/.git
rm -f -r ./vlc/.git ./vl-checklist/.gitignore ./vl-checklist/example_models ./vl-checklist/docs
