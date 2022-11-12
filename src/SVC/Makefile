VERSION := 1
IMAGE := racahu23/svc:${VERSION}
#PLATFORM := intel/x86-64

.PHONY: build push install-devs

build:
	docker buildx build --push --platform  linux/amd64 \
		. \
		-t ${IMAGE}
#--platform ${PLATFORM} \
push:

