const faker = require('faker');

const fields = {
  'address': [
    'zipCode'
    ,'city'
    ,'cityPrefix'
    ,'citySuffix'
    ,'streetName'
    ,'streetAddress'
    ,'streetSuffix'
    ,'streetPrefix'
    ,'secondaryAddress'
    ,'county'
    ,'country'
    ,'countryCode'
    ,'state'
    ,'stateAbbr'
    ,'latitude'
    ,'longitude'
  ],
  'commerce': [
    'color'
    ,'department'
    ,'productName'
    ,'price'
    ,'productAdjective'
    ,'productMaterial'
    ,'product'
  ],
  'company': [
    'suffixes'
    ,'companyName'
    ,'companySuffix'
    ,'catchPhrase'
    ,'bs'
    ,'catchPhraseAdjective'
    ,'catchPhraseDescriptor'
    ,'catchPhraseNoun'
    ,'bsAdjective'
    ,'bsBuzz'
    ,'bsNoun'
  ],
  'database': [
    'column'
    ,'type'
    ,'collation'
    ,'engine'
  ],
  'date': [
    'past'
    ,'future'
    ,'between'
    ,'recent'
    ,'month'
    ,'weekday'
  ],
  'fake': [],
  'finance': [
    'account'
    ,'accountName'
    ,'mask'
    ,'amount'
    ,'transactionType'
    ,'currencyCode'
    ,'currencyName'
    ,'currencySymbol'
    ,'bitcoinAddress'
    ,'iban'
    ,'bic'
  ],
  'hacker': [
    'abbreviation'
    ,'adjective'
    ,'noun'
    ,'verb'
    ,'ingverb'
    ,'phrase'
    ,'helpers'
    ,'randomize'
    ,'slugify'
    ,'replaceSymbolWithNumber'
    ,'replaceSymbols'
    ,'shuffle'
    ,'mustache'
    ,'createCard'
    ,'contextualCard'
    ,'userCard'
    ,'createTransaction'
  ],
  'image': [
    'image'
    ,'avatar'
    ,'imageUrl'
    ,'abstract'
    ,'animals'
    ,'business'
    ,'cats'
    ,'city'
    ,'food'
    ,'nightlife'
    ,'fashion'
    ,'people'
    ,'nature'
    ,'sports'
    ,'technics'
    ,'transport'
    ,'dataUri'
  ],
  'internet': [
    'avatar'
    ,'email'
    ,'exampleEmail'
    ,'userName'
    ,'protocol'
    ,'url'
    ,'domainName'
    ,'domainSuffix'
    ,'domainWord'
    ,'ip'
    ,'ipv6'
    ,'userAgent'
    ,'color'
    ,'mac'
    ,'password'
  ],
  'lorem': [
    'word'
    ,'words'
    ,'sentence'
    ,'slug'
    ,'sentences'
    ,'paragraph'
    ,'paragraphs'
    ,'text'
    ,'lines'
  ],
  'name': [
    'firstName'
    ,'lastName'
    ,'findName'
    ,'jobTitle'
    ,'prefix'
    ,'suffix'
    ,'title'
    ,'jobDescriptor'
    ,'jobArea'
    ,'jobType'
  ],
  'phone': [
    'phoneNumber'
    ,'phoneNumberFormat'
    ,'phoneFormats'
  ],
  'random': [
    'number'
    ,'arrayElement'
    ,'objectElement'
    ,'uuid'
    ,'boolean'
    ,'word'
    ,'words'
    ,'image'
    ,'locale'
    ,'alphaNumeric'
  ],
  'system': [
    'fileName'
    ,'commonFileName'
    ,'mimeType'
    ,'commonFileType'
    ,'commonFileExt'
    ,'fileType'
    ,'fileExt'
    ,'directoryPath'
    ,'filePath'
    ,'semver'
  ]
};

const randint = function (min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min;
}

// https://www.world-airport-codes.com/world-top-30-airports.html
const airports1 = [ 'SEA', 'CTU', 'YYZ', 'MUC', 'BOM', 'FCO'];
const airports2 = ['LGW', 'SYD', 'SZX', 'BCN', 'GRU', 'SHA', 'MCO' ];

const flightCodes = function(airports) {
  return airports[randint(0,airports.length)]
}

const specialRules = function(field) {
  switch (field) {
    case 'origin1':
      return flightCodes(airports1);
      break;
    case 'destination1':
      return flightCodes(airports2);
      break;
    case 'flightDate1':
      return randint(0,3) + '';
      break;
    case 'flightDate2':
      return randint(3,7) + '';
      break;
    case 'flexible':
      return !!(Math.random() > 0.5) + '';
      break;
    case 'travelers':
      return randint(1, 5) + '';
      break;
    default:
      break;
  }
}

module.exports = function(field) {
  for(var name in fields) {
    var category = fields[name];
    for(var i = 0; i < category.length; i++) {
      var f = category[i];
      if(f.toUpperCase() === field.toUpperCase()) {
        // console.log(name, f);
        const faked = faker[name][f]()
        // console.log(faked);
        return faked;
      }
    }
  }
  const special = specialRules(field);
  if (special !== null) return special;
  console.warn(`No generator found for "${field}"`);
  return 'UNK';
}

