//! Compile kernel `lib/math/int_sqrt.c` as userspace and check floor(sqrt(x)).

use std::fs;
use std::process::Command;

fn extract_int_sqrt(src: &str) -> &str {
    let start = src
        .find("unsigned long int_sqrt(unsigned long x)")
        .expect("int_sqrt definition");
    let body = &src[start..];
    let end = body
        .find("EXPORT_SYMBOL(int_sqrt)")
        .expect("EXPORT_SYMBOL(int_sqrt)");
    body[..end].trim()
}

fn userspace_program(func: &str) -> String {
    format!(
        r#"
#include <stdio.h>

static unsigned long __fls(unsigned long word)
{{
    return word ? (8 * sizeof(unsigned long) - 1 - (unsigned long)__builtin_clzl(word)) : 0;
}}

{func}

int main(void)
{{
    struct {{ unsigned long in; unsigned long out; }} cases[] = {{
        {{0, 0}}, {{1, 1}}, {{2, 1}}, {{3, 1}}, {{4, 2}}, {{8, 2}}, {{9, 3}},
        {{15, 3}}, {{16, 4}}, {{24, 4}}, {{25, 5}}, {{99, 9}}, {{100, 10}},
        {{101, 10}}, {{255, 15}}, {{256, 16}}
    }};
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {{
        unsigned long got = int_sqrt(cases[i].in);
        if (got != cases[i].out) {{
            fprintf(stderr, "int_sqrt(%lu) = %lu, expected %lu\n",
                    cases[i].in, got, cases[i].out);
            return 1;
        }}
    }}
    return 0;
}}
"#
    )
}

#[test]
fn kernel_int_sqrt_matches_floor_sqrt() {
    let src = fs::read_to_string("lib/math/int_sqrt.c").expect("lib/math/int_sqrt.c");
    assert!(
        src.contains("unsigned long int_sqrt(unsigned long x)"),
        "int_sqrt missing from lib/math/int_sqrt.c"
    );
    let func = extract_int_sqrt(&src);
    let program = userspace_program(func);
    let dir = std::env::temp_dir().join(format!("linux-int-sqrt-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let c_path = dir.join("int_sqrt_test.c");
    let bin_path = dir.join("int_sqrt_test");
    fs::write(&c_path, program).unwrap();
    let compile = Command::new("cc")
        .arg(c_path.as_os_str())
        .arg("-o")
        .arg(bin_path.as_os_str())
        .output()
        .expect("spawn cc");
    assert!(
        compile.status.success(),
        "cc failed:\n{}\n{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&bin_path).output().expect("run int_sqrt_test");
    assert!(
        run.status.success(),
        "int_sqrt cases failed:\n{}\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let _ = fs::remove_dir_all(dir);
}
