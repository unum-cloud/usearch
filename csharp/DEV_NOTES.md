## Build and Test

1. **Locate the Runtime Identifier (RID):** Determine the RID corresponding to your system, such as `linux-x64` or `win-x64`.
2. **Place the Dynamic Library:** Copy the `libusearch_c` dynamic library to the `src/LibUSearch.Tests/runtimes/{RID}/native/` directory, replacing `{RID}` with the value from step 1.
3. **Run the Script from Root Directory:** Execute the `build_and_test.sh` script with the command `./build_and_test.sh`.
