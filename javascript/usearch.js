const usearch_native = require('bindings')('usearch');

class Index {
    /**
     * Creates an instance of Index.
     * @date 6/7/2023 - 7:32:29 AM
     *
     * @constructor
     * @param {...{}} args
     */
    constructor(...args) {
        this.native = new usearch_native.Index(...args)
    }

    /**
     * Description placeholder
     * @date 6/7/2023 - 7:32:28 AM
     *
     * @param {...{}} args
     * @returns {*}
     */
    dimensions(...args) {
        return this.native.dimensions(...args)
    }

    /**
     * Description placeholder
     * @date 6/7/2023 - 7:32:28 AM
     *
     * @param {...{}} args
     * @returns {*}
     */
    size(...args) {
        return this.native.size(...args)
    }

    /**
     * Description placeholder
     * @date 6/7/2023 - 7:32:28 AM
     *
     * @param {...{}} args
     * @returns {*}
     */
    capacity(...args) {
        return this.native.capacity(...args)
    }

    /**
     * Description placeholder
     * @date 6/7/2023 - 7:32:28 AM
     *
     * @param {...{}} args
     * @returns {*}
     */
    connectivity(...args) {
        return this.native.connectivity(...args)
    }

    /**
     * Description placeholder
     * @date 6/7/2023 - 7:32:28 AM
     *
     * @param {...{}} args
     * @returns {*}
     */
    save(...args) {
        return this.native.save(...args)
    }

    /**
     * Description placeholder
     * @date 6/7/2023 - 7:32:27 AM
     *
     * @param {...{}} args
     * @returns {*}
     */
    load(...args) {
        return this.native.load(...args)
    }

    /**
     * Description placeholder
     * @date 6/7/2023 - 7:32:27 AM
     *
     * @param {...{}} args
     * @returns {*}
     */
    view(...args) {
        return this.native.view(...args)
    }

    /**
     * Description placeholder
     * @date 6/7/2023 - 7:32:27 AM
     *
     * @param {...{}} args
     * @returns {*}
     */
    add(...args) {
        return this.native.add(...args)
    }

    /**
     * Description placeholder
     * @date 6/7/2023 - 7:32:27 AM
     *
     * @param {...{}} args
     * @returns {*}
     */
    search(...args) {
        return this.native.search(...args)
    }
}


module.exports = {
    Index: Index
};