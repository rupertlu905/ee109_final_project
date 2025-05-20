You can run Scalaism on the example Spatial code with the following command:
```
sbt -Dtest.CS217=true "; testOnly HelloWorld"
```

You can run VCS simulation on the example Spatial code with the following command:
```
source exports.sh
sbt -Dtest.VCS=true "; testOnly HelloWorld"
```
