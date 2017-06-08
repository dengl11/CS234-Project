const fs = require('fs');
const print = console.log

function getTerm(frag) {
  return frag.substring(frag.indexOf("$")+1,frag.lastIndexOf("$"));
}

function ds(slot, str) {
  const pieces = str.split('|')
  const ds = {}
  ds[slot] = '';
  for (var i = 0; i < pieces.length; i++) {
    var term = getTerm(pieces[i]);
    if (term) {
      ds[term] = pieces[i].trim();
    } else {
      ds[slot] += pieces[i].trim();
    }
  }
  return ds;
}

function combos(a, min) {
    var fn = function(n, src, got, all) {
        if (n == 0) {
            if (got.length > 0) {
                all[all.length] = got;
            }
            return;
        }
        for (var j = 0; j < src.length; j++) {
            fn(n - 1, src.slice(j + 1), got.concat([src[j]]), all);
        }
        return;
    }
    var all = [];
    for (var i = min; i < a.length; i++) {
        fn(i, a, [], all);
    }
    all.push(a);
    return all;
}


var obj = JSON.parse(fs.readFileSync('dia_act_nl_pairs.flights.json', 'utf8'));


agt = {};
usr = {};
const stuff = obj.dia_acts.request.map(da => {
  const rs = da.request_slots[0]
  return {
    usr: [rs, ds(rs, da.nl.usr)],
    agt: [rs, ds(rs, da.nl.agt)]
  }
}).reduce((c, o) => {
  c.usr.push({request: o.usr[0], inform: o.usr[1]});
  c.agt.push({request: o.agt[0], inform: o.agt[1]});
  return c;
}, {usr: [], agt: []});

function vecify(map) {
  var out = []
  for(var k in map) {
    out.push([k, map[k]]);
  }
  return out;
}

function domagic(utype, reqtypes, reqfrag, cs) {
  magic = (options) => {
    let sentence = []
    let informs = []
    let out = {nl: {}}
    sentence.push(reqfrag)
    options.map(foolishness => {
      sentence.push(foolishness[1])
      informs.push(foolishness[0])
    })
    sentence.splice(sentence.length-1, 0, 'and');
    fullsentence = sentence.join(' ') + '?'
    out.request_slots = reqtypes
    out.nl[utype] = fullsentence
    out.inform_slots = informs
    return out
  }
  return cs.map(magic)
}

// the code you write when you've given up on everything.
let out = stuff.usr.map(thing => {
  var reqtype = thing.request
  var reqfrag = thing.inform[reqtype]
  var newthing = Object.assign(thing.inform)
  delete newthing[reqtype]
  var vecthing = vecify(newthing)
  return domagic('usr', [reqtype], reqfrag, combos(vecthing, 2))
});
let out2 = stuff.agt.map(thing => {
  var reqtype = thing.request
  var reqfrag = thing.inform[reqtype]
  var newthing = Object.assign(thing.inform)
  delete newthing[reqtype]
  var vecthing = vecify(newthing)
  return domagic('agt', [reqtype], reqfrag, combos(vecthing, 2))
});

for(var i = 0; i < out.length; i++) {
  informu = out[i];
  informa = out2[i];
  for(var j = 0; j < informu.length; j++) {
    out[i][j].nl.agt = out2[i][j].nl.agt;
  }
}

out = out.reduce((coll, nxt) => {
  return coll.concat(nxt)
}, [])

out = {dia_acts: {request: out}}

print(
  JSON.stringify(out,null,2)
)
