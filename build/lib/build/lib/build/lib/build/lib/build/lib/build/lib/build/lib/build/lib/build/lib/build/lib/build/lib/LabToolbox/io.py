def to_latex_table(data, header, filename, caption, label):
    """
    Writes a LaTeX-formatted table to file with caption, label and predefined styling.

    Parameters
    ----------
    data : list of list of str or float
        The content of the table, organized as a list of rows.
    header : list of str
        List of column names to appear in the header of the table.
    filename : str
        Path to the output `.tex` file (e.g., 'table.tex').
    caption : str
        Caption text of the table.
    label : str
        Label used for referencing the table in LaTeX.
    
    Notes
    -----
    - Assumes that all elements of `data` and `header` are convertible to string.
    - Does not escape LaTeX special characters: input should already be LaTeX-safe.
    """

    n_cols = len(header)
    col_format = "c" * n_cols

    with open(filename, 'w') as f:
        f.write("\\begin{table}[H]\n")
        f.write(f"\\caption{{\\large \\label{{{label}}} {caption}}}\n")
        f.write("\\vspace{-0.7\\baselineskip}\n")
        f.write("\\centering\n")
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\hline\\hline\n")
        f.write("\\noalign{\\vskip 1.5pt}\n")
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\noalign{\\vskip 2pt}\n")

        for row in data:
            row_str = " & ".join(str(cell) for cell in row)
            f.write(row_str + " \\\\\n")

        f.write("\\noalign{\\vskip 1.5pt}\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")