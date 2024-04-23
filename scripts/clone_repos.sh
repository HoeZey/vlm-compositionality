git clone https://github.com/mertyg/vision-language-models-are-bows.git aro
git clone https://github.com/om-ai-lab/VL-CheckList.git vlc
git clone https://github.com/RAIVNLab/sugar-crepe.git 

# Remove unnecessary SugarCrepe
rm -f -r ./sugar-crepe/.git ./sugar-crepe/.gitignore ./sugar-crepe/assets ./sugar-crepe/gpt-4v-results

# Remove unnecessary ARO files
rm -f -r ./aro/.git aro/.gitignore ./aro/scripts ./aro/temp_data 

# Remove unnecessary VL-CheckList files
rm -f -r ./vl-checklist/.git ./vl-checklist/.gitignore ./vl-checklist/example_models ./vl-checklist/docs

# Move repos to benchmark folder
rm -r ../benchmarks/*
mv sugar-crepe ../benchmarks
mv aro ../benchmarks
mv vl-checklist ../benchmarks
