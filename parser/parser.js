const cheerio = require('cheerio');
const request = require('request-promise');
const fakeit = require('./fakeit');
const fs = require('fs');


/* ==== fetch JSON from HTML form ==== */
const arr2json = (arr) => {
  return arr.reduce((out, field) => {
    out[field.name] = field.value;
    return out;
  }, {});
}

const html2json = ($) => {
  arrs = [];
  $('form').each((i, elem) => {
    var $form = $('<form>');

    $form.append($(elem).find('input'));

    let fs = '<form>' + $form.html() + '</form>';
    $$ = cheerio.load(fs);

    arrs.push($$('form').serializeArray());
  });

  return arrs.map(arr2json);
};

/* ==== convert raw json to goal ==== */
const json2goal = (json) => {
  request_slots = {};
  inform_slots = {};
  diaact = 'request';
  for(const key in json) {
    let value = json[key];
    // if (!value) { value = fakeit(key); }
    // value = fakeit(key);
    if (value === '') {
      request_slots[key] = 'UNK';
    } else {
      inform_slots[key] = value;
    }
  }
  return Object.keys(json).length > 0 ? {request_slots, inform_slots, diaact} : null;
}



/********   Parse URL/HTML   ********/
/*
  return: promise of parsed result
*/

exports.parse = (options) => {
  const {url, html} = options;

  let handler;

  if(url) {
    handler = request(url)
  }
  if(html) {
    handler = Promise.resolve(html);
  }

  return handler
    .then(htmlString => cheerio.load(htmlString))
    .then(html2json)
    .then(jsons => jsons.map(json2goal))
    .then(tasks => tasks.filter(t => t))
    .then(console.log)
    .catch(err => {
      console.error('Crawling Failed');
      console.error(err);
    });
}

function repeatSimple(element, times)
{
    var result = [];
    for(var i=0;i<times;i++)
      result.push(element);
    return result;
}

/* ==== convert raw json to kb ==== */
const populateJSON = (json) => {
  for(const key in json) {
    json[key] = fakeit(key);
  }
  return json;
}

const arr2kb = (kb, entry, i) => {
  kb[i+''] = entry;
  return kb;
}

const write = (filename, data) => {
  fs.writeFileSync(filename, JSON.stringify(data));
  return data;
}

const randomSubarray = (arr, size) => {
    var shuffled = arr.slice(0), i = arr.length, min = i - size, temp, index;
    while (i-- > min) {
        index = Math.floor((i + 1) * Math.random());
        temp = shuffled[index];
        shuffled[index] = shuffled[i];
        shuffled[i] = temp;
    }
    return shuffled.slice(min);
}

const kb2goals = (kb, size) => {
  goalmaterial = randomSubarray(kb, size);
  return goalmaterial.map(json2goal);
}

exports.load = (array, options) => {
  return Promise.resolve(repeatSimple(array, 1000))
    .then(arrs => arrs.map(arr2json))
    .then(jsons => jsons.map(populateJSON))
    .then(arr => {
      const kb = arr.reduce(arr2kb, {});
      const goals = kb2goals(arr, 128)

      write(options.kbpath, kb);
      write(options.goalpath, goals);
      return kb
    })
    .then(console.log)
    .catch(err => {
      console.error('Loading Failed');
      console.error(err);
    });
}

