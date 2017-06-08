#!/usr/bin/env node

const parser = require('./parser');

const argv = require('yargs').help().argv;

const fs = require('fs');

output_path = argv.output
if(!output_path) {
	console.error("Please specify output file path: --output")
	return;
}

/***** Parsing HTML *****/
if(argv.html) {
	parser.parse({html: argv.html});
}

/***** Parsing File *****/
if(argv.file) {
  const html = fs.readFileSync(argv.file, 'utf8');
  parser.parse({html});
}

/***** Parsing File *****/
if(argv.load) {
  const arr = JSON.parse(fs.readFileSync(argv.load, 'utf8'));
  parser.load(arr, {kbpath: argv.kbpath, goalpath: argv.goalpath});
}

/***** Parsing URL and Dump to File *****/
if(argv._[0] || argv.url) {
  const url = argv.url || argv._[0]
  parsed_handler = parser.parse({url: url});

  parsed_handler
  // .then(parsed => fs.writeFileSync("./goal.json", JSON.stringify(parsed)))
  .then(console.log("URL Parsed:        ", url))
  .then(console.log("Result written to: ", output_path))
  .catch(err => {
    console.error('File Dumping Failed');
    console.error(err);
  });
}
