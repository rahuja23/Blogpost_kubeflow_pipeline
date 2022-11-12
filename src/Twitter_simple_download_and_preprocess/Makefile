VERSION := 1
#IMAGE := datareply/ml-blueprint_dataloader:${VERSION}
IMAGE := racahu23/blogpost-twitter-download:${VERSION}
#PLATFORM := intel/x86-64

.PHONY: build push install-devs

build:
	docker buildx build --push --platform  linux/amd64 \
		. \
		-t ${IMAGE}
#--platform ${PLATFORM} \
push:

