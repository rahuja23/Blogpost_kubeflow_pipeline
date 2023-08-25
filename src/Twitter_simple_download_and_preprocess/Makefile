<<<<<<< HEAD
VERSION := 4
=======
VERSION := 1
#IMAGE := datareply/ml-blueprint_dataloader:${VERSION}
>>>>>>> 9643c8dbf8c807e406368a719a7a61d4b9b6399e
IMAGE := racahu23/blogpost-twitter-download:${VERSION}
#PLATFORM := intel/x86-64

.PHONY: build push install-devs

build:
	docker buildx build --push --platform  linux/amd64 \
		. \
		-t ${IMAGE}
#--platform ${PLATFORM} \
push:

