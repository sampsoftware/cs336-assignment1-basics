# Essay Answers



## Problem (unicode1): Understanding Unicode (1 point)

### (a)
What Unicode character does chr(0) return?
Deliverable: A one-sentence response.

'\x00' - Hex zero, unicode codepoint for null.

### (b)
How does this character’s string representation (__repr__()) differ from its printed representation?
Deliverable: A one-sentence response.

The string representation is a string of the hex notation for the null character, while the character itself, when printed, is a null.

### (c)
What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
Deliverable: A one-sentence response.

It prints nothing - no character at all.

## Problem (unicode2): Unicode Encodings (3 points)
### (a)
What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
Deliverable: A one-to-two sentence response.

UTF8 is 1 to 4 8-bit bytes, and UTF16 is 1 to 2 16-bit word. UTF32 is always one 32-bit word. UTF8 generally places ASCII charaters in the low range, so only one byte is generally required. If UTF16 were used, the minimum would be 16 bytes; UTF32 is even worse. In the exception case where there's a more-than-one-byte UTF8, it still fits. 



### (b)
Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
Deliverable: An example input byte string for which decode_utf8_bytes_to_str_wrong produces incorrect output, with a one-sentence explanation of why the function is incorrect.

The function does not work because it expects each unicode codepoint to be one byte, but some are more than one byte. こんにちは would not work

### (c)
Give a two-byte sequence that does not decode to any Unicode character(s).
Deliverable: An example, with a one-sentence explanation.

\x00\x10 does not; the bits of the first byte are unused. 
\x10\x00 does not; the bits of the second byte are unused and UTF8 only provides significant bytes.

