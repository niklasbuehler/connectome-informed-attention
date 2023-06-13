#!/bin/bash

while true; do
	echo "installing conda ..."
	if mkdir -p ~/miniconda3 ; then
	    echo -e "\e[92mConda 1!\e[0m"
	else
	    echo -e "\e[91mFailed to create folder\e[0m"
	    break
	fi

	if wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh ; then
	    echo -e "\e[92mConda 2!\e[0m"
	else
	    echo -e "\e[91mFailed to create folder\e[0m"
	    break
	fi

	if bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 ; then
	    echo -e "\e[92mConda 3!\e[0m"
	else
	    echo -e "\e[91mFailed to create folder\e[0m"
	    break
	fi

	if rm -rf ~/miniconda3/miniconda.sh ; then
	    echo -e "\e[92mConda 4!\e[0m"
	else
	    echo -e "\e[91mFailed to create folder\e[0m"
	    break
	fi

	if ~/miniconda3/bin/conda init bash ; then
	    echo -e "\e[92mConda 5!\e[0m"
	else
	    echo -e "\e[91mFailed to create folder\e[0m"
	    break
	fi

	if ~/miniconda3/bin/conda init zsh ; then
	    echo -e "\e[92mConda 6!\e[0m"
	else
	    echo -e "\e[91mFailed to create folder\e[0m"
	    break
	fi

	break
done
	    echo -e "\e[92mEnvirnment ready!\e[0m"
