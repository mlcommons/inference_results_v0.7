if [ ! -d quantized_data/to_compile ]; then
	bash quantize.sh
fi

if [ ! -f resnet50.tpu ]; then
	.venv/bin/tcf compile --optimize quantized_data/to_compile resnet50.tpu
fi

echo "compiled successfully"
