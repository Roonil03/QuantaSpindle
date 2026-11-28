package main

import (
	"flag"
	"strconv"
	"strings"
)

// scanJSONErrors discovers the opt-in even after a flag that will fail parsing.
// It follows flag's token boundaries: registered nonboolean flags consume their
// next token as a value, and -- or a positional argument ends the scan. Unknown
// flags consume only their own token because they have no registered value type.
func scanJSONErrors(args []string, fs *flag.FlagSet) bool {
	enabled := false
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" || arg == "-" || !strings.HasPrefix(arg, "-") {
			break
		}
		name := strings.TrimPrefix(arg, "-")
		name = strings.TrimPrefix(name, "-")
		name, value, assigned := strings.Cut(name, "=")
		if name == "json-errors" {
			if !assigned {
				enabled = true
			} else if parsed, err := strconv.ParseBool(value); err == nil {
				enabled = parsed
			}
			continue
		}
		if option := fs.Lookup(name); option != nil && !assigned {
			if boolean, ok := option.Value.(interface{ IsBoolFlag() bool }); ok && boolean.IsBoolFlag() {
				continue
			}
			i++
		}
	}
	return enabled
}
