# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* paittern/*.py

black:
	@black scripts/* paittern/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr paittern-*.dist-info
	@rm -fr paittern.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# project id - replace with your GCP project id
PROJECT_ID=wagon-bootcamp-336718

# bucket name - replace with your GCP bucket name
BUCKET_NAME=wagon-bootcamp-paittern
BUCKET_FOLDER=people_segmentation
BUCKET_FOLDER2=augmented_data
LOCAL_PATH = /Users/humbert/Documents/Human-Image-Segmentation-with-DeepLabV3Plus-in-TensorFlow-main/people_segmentation/*
LOCAL_PATH2 = /Users/humbert/Documents/Human-Image-Segmentation-with-DeepLabV3Plus-in-TensorFlow-main/new_data/*

upload_data:
	-@gsutil cp -r ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/

upload_data2:
	-@gsutil -m cp -r ${LOCAL_PATH2} gs://${BUCKET_NAME}/${BUCKET_FOLDER2}/


# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

PACKAGE_NAME = paittern
FILENAME = contouring.trainer2
JOB_NAME=contouring_$(shell date +'%Y%m%d_%H%M%S')
BUCKET_TRAINING_FOLDER = 'trainings'
PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=2.2

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--parameter-server-accelerator=count=60,type=nvidia-tesla-p100 \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs