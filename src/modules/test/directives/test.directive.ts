import { Directive, Input } from '@angular/core';

@Directive({
    selector: '[sbTest]',
})
export class TestDirective {
    @Input() param!: string;

    constructor() {}
}
