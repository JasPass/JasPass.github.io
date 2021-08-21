import { ChangeDetectionStrategy, Component, OnInit } from '@angular/core';

@Component({
    selector: 'sb-test',
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './test.component.html',
    styleUrls: ['test.component.scss'],
})
export class TestComponent implements OnInit {
    constructor() {}
    ngOnInit() {}
}
