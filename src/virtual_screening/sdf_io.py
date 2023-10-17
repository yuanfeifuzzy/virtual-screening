#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
from pathlib import Path

from rdkit import Chem


class SDF:
    def __init__(self, s, deep=False):
        self.s = s
        self.title, self.mol, self.properties, self.properties_dict = self.parse(s, deep=deep)

    @staticmethod
    def parse(s, deep=False):
        if s:
            lines = s.splitlines(keepends=True)
            title = lines[0].rstrip()

            mol, i = [], 0
            for i, line in enumerate(lines[1:]):
                mol.append(line)
                if line.strip() == 'M  END':
                    break
            mol = ''.join(mol)

            properties = lines[i + 2:]

            properties_dict, idx = {}, []
            if deep:
                for i, p in enumerate(properties):
                    if p.startswith('>'):
                        properties_dict[p.split(maxsplit=1)[1].rstrip()] = ''
                        idx.append(i)
                idx.append(-1)

                for i, k in enumerate(properties_dict.keys()):
                    properties_dict[k] = ''.join(properties[idx[i] + 1:idx[i + 1]])

            properties = ''.join(properties[:-1])
        else:
            title, mol, properties, properties_dict = '', '', '', {}
        return title, mol, properties, properties_dict

    def sdf(self, output=None, properties=False, title=''):
        if properties:
            s = f'{title or self.title}\n{self.mol}\n{self.properties}\n$$$$\n'
        else:
            s = f'{title or self.title}\n{self.mol}\n$$$$\n'
        if output:
            with open(output, 'w') as o:
                o.write(s)
        return s

    def __str__(self):
        return self.sdf()


def parse(sdf, string=False, deep=False):
    opener = gzip.open if str(sdf).endswith('.gz') else open
    with opener(sdf, 'rt') as f:
        lines = []
        for i, line in enumerate(f):
            lines.append(line)
            if line.strip() == '$$$$':
                yield ''.join(lines) if string else SDF(''.join(lines), deep=deep)
                lines = []
                continue
        yield ''.join(lines) if string else SDF(''.join(lines), deep=deep)


def write(ss, output, properties=False):
    with open(output, 'w') as o:
        o.writelines(f'{s.sdf(properties=properties) if isinstance(s, SDF) else s}\n$$$$\n' for s in ss)
    return output