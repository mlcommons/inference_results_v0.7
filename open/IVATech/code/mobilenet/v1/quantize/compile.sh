if [ ! -d quantized_data/to_compile ]; then
	bash quantize.sh
fi

if [ ! -f mobilenet.tpu ]; then
	.venv/bin/tcf compile --optimize quantized_data/to_compile mobilenet.tpu
fi

echo "compiled successfully"
