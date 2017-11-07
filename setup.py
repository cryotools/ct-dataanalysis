from setuptools import setup, find_packages
PACKAGES = find_packages()

opts = dict(name="ct-dataanalysis",
            maintainer="David Loibl",
            maintainer_email="info@davidloibl.de",
            description="A collection of Python scripts for basic statistic analysis and plotting.",
            long_description="A collection of Python scripts for basic statistic analysis and plotting.",
            url="https://cryo-tools.org/tools/ct-statistics/",
            download_url="https://github.com/cryotools/ct-dataanalysis.git",
            license="BSD-3-Clause",
            packages=PACKAGES)


if __name__ == '__main__':
    setup(**opts)
