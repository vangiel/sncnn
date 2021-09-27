for FILE in images_dataset/*.json
do
  echo "$FILE"
  IFS='/' read -ra ADDR <<< "$FILE"
  IFS='.' read -ra ADDR <<< "${ADDR[1]}"
  cp "../sngnnv2/unlabelled_data/${ADDR[0]}.mp4" images_dataset/
done