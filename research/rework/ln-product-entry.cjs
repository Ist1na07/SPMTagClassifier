'use strict';
// Pure exports for the static product build; dynamic archive access is
// replaced by scripts/build-ln.cjs before this entry is shipped.
module.exports={...require('./ln-adapter-direct.cjs'),nativeEvidence:require('./evidence.cjs').evidence,
 classes:require('./native.cjs').CLASSES};
