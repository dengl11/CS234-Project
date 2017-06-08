const submittableSelector = 'input,select,textarea,keygen';
const r20 = /%20/g;
const rCRLF = /\r?\n/g;
const _ = require('lodash');

exports.serializeArray = function(Cheerio, el) {
  return el.map(function() {
    var elem = el;
    var $elem = Cheerio(elem);
    if (elem.name === 'form') {
      return $elem.find(submittableSelector).toArray();
    } else {
      return $elem.filter(submittableSelector).toArray();
    }
  }).filter(
  // Verify elements have a name (`attr.name`) and are not disabled (`:disabled`)
  '[name!=""]:not(:disabled)'
  // and cannot be clicked (`[type=submit]`) or are used in `x-www-form-urlencoded` (`[type=file]`)
  + ':not(:submit, :button, :image, :reset, :file)'
  // and are either checked/don't have a checkable state
  + ':matches([checked], :not(:checkbox, :radio))'
  // Convert each of the elements to its value(s)
  ).map(function(i, elem) {
    var $elem = Cheerio(elem);
    var name = $elem.attr('name');
    var value = $elem.val();

    // If there is no value set (e.g. `undefined`, `null`), then default value to empty
    if (value == null) {
      value = '';
    }

    // If we have an array of values (e.g. `<select multiple>`), return an array of key/value pairs
    if (Array.isArray(value)) {
      return _.map(value, function(val) {
        // We trim replace any line endings (e.g. `\r` or `\r\n` with `\r\n`) to guarantee consistency across platforms
        //   These can occur inside of `<textarea>'s`
        return {name: name, value: val.replace( rCRLF, '\r\n' )};
      });
      // Otherwise (e.g. `<input type="text">`, return only one key/value pair
    } else {
      return {name: name, value: value.replace( rCRLF, '\r\n' )};
    }
    // Convert our result to an array
  }).get();
}
