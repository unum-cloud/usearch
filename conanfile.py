from conans import ConanFile


class USearchConan(ConanFile):

    name = 'usearch'
    version = '2.4.0'
    license = 'Apache 2.0'
    url = 'https://github.com/conan-io/conan-center-index'
    description = 'Smaller & Faster Single-File Vector Search Engine from Unum'
    homepage = 'https://github.com/unum-cloud/usearch'
    topics = 'vector-search'

    # No settings/options are necessary, this is header only
    # Potentially add unit-tests in the future:
    # https://docs.conan.io/1/howtos/header_only.html#with-unit-tests
    exports_sources = 'include/*'
    no_copy_source = True

    def package(self):
        self.copy('*.h')
