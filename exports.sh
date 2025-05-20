# export _JAVA_OPTIONS="-Xmx64g -Xss8912k -Xms16g -XX:ReservedCodeCacheSize=2G"
# export _JAVA_OPTIONS="-Dsbt.ivy.home=$HOME/.sbt/.ivy2 -Dsbt.boot.directory=$HOME/.sbt/boot/ $_JAVA_OPTIONS"
# export SPATIAL_HOME=$HOME/spatial

export VCS_HOME=/cad/synopsys/vcs/K-2015.09-SP2-7
export LM_LICENSE_FILE=27000@cadlic0.stanford.edu:$LM_LICENSE_FILE
export PATH=/cad/synopsys/pts/M-2017.06-SP3/bin:/cad/synopsys/icc/M-2016.12-SP2/bin:/cad/synopsys/dc_shell/latest/bin:$HOME/sbt/bin:$VCS_HOME/amd64/bin:/cad/synopsys/vcs/K-2015.09-SP2-7/bin:$PATH

# export TEST_DATA_HOME=/home/mattfel/test-data/
# export NOVA_HOME="$HOME/autograder/spatial/"

export XILINX_VIVADO=/opt/Xilinx/Vivado/2017.1
export PATH=$XILINX_VIVADO/bin:$PATH
# export PATH=/usr/bin:/local/ssd/home/mattfel/aws-fpga/hdk/common/scripts:$PATH

export LM_LICENSE_FILE=7193@cadlic0.stanford.edu:$LM_LICENSE_FILE
# export LM_LICENSE_FILE=/opt/Xilinx/awsF1.lic:7193@cadlic0.stanford.edu:$LM_LICENSE_FILE

export CLOCK_FREQ_MHZ="100"

export NUM_THREADS="10"

export PATH="/opt/Xilinx/SDK/2017.1/bin/":$PATH
export PATH="/local/ssd/opt/Xilinx/Vivado/2019.1/bin/:$PATH"
