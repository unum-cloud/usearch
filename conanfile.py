from conans import ConanFile


class USearchConan(ConanFile):

    name = 'USearch'
    version = '2.0.2'
    license = 'Apache 2.0'
    url = 'https://github.com/unum-cloud/usearch'
    description = 'Smaller & Faster Single-File Vector Search Engine from Unum'

    # No settings/options are necessary, this is header only
    # Potentially add unit-tests in the future:
    # https://docs.conan.io/1/howtos/header_only.html#with-unit-tests
    exports_sources = 'include/*'
    no_copy_source = True

    def package(self):
        self.copy('*.h')
